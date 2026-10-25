# Pj1: Perception System Concept and Design (Deteksi Bola dan Ring untuk MicroDuck)

**Mata kuliah:** RET503 Computer Vision and Deep Learning
**Nama / NIM:** Zahra Nabila Putri / 4222401028
**Kelompok:** «ISI nama kelompok dan anggota»

Projek ini merancang sistem persepsi untuk robot MicroDuck pada tugas robot basket:
dari penentuan lingkup persepsi, target kinerja, dan pipeline, sampai dataset YOLO
dan baseline YOLO yang dilatih dengan log eksperimen.

## 1. Lingkup persepsi

Robot harus mendeteksi **bola** (`ball`) sebagai objek yang ditendang dan **ring** (`rim`)
sebagai target arah tendangan, memakai satu kamera. Kelas `backboard` ikut dideteksi sebagai
konteks posisi. Keluaran persepsi per objek berupa kelas, kotak, dan confidence. Posisi
horizontal kotak dipakai untuk mengarahkan badan robot, ukuran kotak sebagai perkiraan jarak,
dan posisi bola terhadap zona tendang sebagai syarat menendang.

| Id | Kelas | Peran |
|---|---|---|
| 0 | backboard | konteks posisi papan |
| 1 | ball | objek yang ditendang |
| 2 | rim | target arah tendangan |

## 2. Target kinerja

| Metrik | Target | Hasil | Status |
|---|---|---|---|
| mAP50-95 (test) | ≥ 0,50 | «ISI» | «tercapai / belum» |
| mAP50 (test) | ≥ 0,85 | «ISI» | «tercapai / belum» |
| AP `ball` dan `rim` (mAP50-95) | masing-masing ≥ 0,40 | «ISI» | «tercapai / belum» |
| Latensi per frame (Orange Pi 3B) | ≤ 100 ms | «belum diukur / ISI» | «ISI» |
| FPS (Orange Pi 3B) | ≥ 10 | «belum diukur / ISI» | «ISI» |

Target ditetapkan sebagai perkiraan awal berdasarkan misi robot: bola bergerak lambat,
sehingga jeda keputusan sekitar 100 ms masih dapat diterima. Alasan perubahan target
(jika ada): «ISI».

## 3. Rancangan pipeline

| Tahap | Pengaturan |
|---|---|
| Kamera | Logitech USB, capture 640×480, 30 fps, format MJPG |
| Posisi kamera | «ISI tinggi dan sudut pemasangan di robot» |
| Praproses | letterbox (resize dengan menjaga rasio) ke 416×416 atau 640×640 |
| Model | YOLO nano (`«yolo26n / yolo11n»`) |
| Perangkat | Orange Pi 3B (Rockchip RK3566, NPU) |
| Pascaproses | NMS, keluaran kelas, confidence, cx, cy, w, h |
| Aksi | kotak dikirim ke logika MicroDuck |

Perkiraan anggaran latensi (belum terukur): ambil frame 10–30 ms, praproses 5–10 ms,
inferensi 40–60 ms, NMS dan keputusan sekitar 5 ms, total sekitar 60–100 ms.
Ukuran input saat deploy harus sama dengan ukuran saat latih, karena model yang
diekspor ke NPU berukuran input tetap.

## 4. Dataset

### Sumber
Frame penuh diambil dari kamera e-con See3CAM_CU135 di kepala robot, di lab BRAIL
pada 24 September 2026, cahaya netral, satu rekaman, setiap 0,5 detik, total 181 frame
(640×480). Frame dianotasi di Roboflow dengan tiga kelas. Sumber berkas frame dan label
mentah: repo `AkbarL090/ret503-transfer-learning`, folder `sumber_frame/`.
Perekam dan penganotasi asli: «ISI nama, dan konfirmasi bahwa data ini data bersama
kelompok». Kamera yang akan dipakai robot (Logitech) berbeda dari kamera perekam.

### Split anti-leakage
Frame diambil berurutan dengan jeda 0,5 detik sehingga frame bertetangga hampir sama.
Karena itu split dibuat berdasarkan **blok frame berurutan** dengan frame jeda di batas
blok, bukan acak per frame. Split ini sama dengan split pada tugas klasifikasi sebelumnya
(`dataset_raw/metadata.csv`). 8 frame jeda (1654–1657 dan 1681–1684) tidak dipakai.

| Split | Frame | backboard | ball | rim |
|---|---|---|---|---|
| train | 124 | 110 | 117 | 115 |
| val | 23 | 20 | 15 | 21 |
| test | 26 | 25 | 26 | 25 |

### Koreksi label
Tiga kotak di data train ternyata salah kelas dan sudah dikoreksi (konsisten dengan
catatan salah label pada tugas klasifikasi):

| Frame | Kesalahan | Koreksi |
|---|---|---|
| 1533 | papan putih berlabel `rim` | menjadi `backboard` |
| 1550 | bola berlabel `rim` | menjadi `ball` |
| 1552 | bola berlabel `rim` | menjadi `ball` |

Pemeriksaan label hanya dilakukan pada tiga kasus ini. Kesalahan label lain yang
belum terdeteksi masih mungkin ada.

### Format
Format YOLO: `images/{train,val,test}` dan `labels/{train,val,test}`, satu berkas `.txt`
per gambar dengan baris `kelas cx cy w h` (ternormalisasi). Konfigurasi ada di `data.yaml`.
Dataset lengkap ada di `dataset_v1.zip`.

## 5. Eksperimen

Data, jumlah epoch (30), batch (16), dan seed (0) sama di semua eksperimen. Setiap
eksperimen hanya mengubah satu faktor. Pelatihan dilakukan di Google Colab (GPU T4)
memakai Ultralytics. Notebook: `colab/Pj1_YOLO_Colab.ipynb`.

| Eksperimen | Perubahan | imgsz | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|---|
| E1_pretrained | baseline, bobot pretrained COCO | 416 | «ISI» | «ISI» | «ISI» | «ISI» |
| E2_scratch | bobot acak (tanpa transfer learning) | 416 | «ISI» | «ISI» | «ISI» | «ISI» |
| E3_tuning | pretrained, ukuran input diubah | 640 | «ISI» | «ISI» | «ISI» | «ISI» |

Metrik di atas dihitung pada data **val**. Berkas log lengkap: `hasil/log_eksperimen.csv`.
Waktu inferensi di GPU Colab hanya pembanding antar eksperimen dan **bukan** hasil
FPS untuk Orange Pi.

AP per kelas (mAP50-95, val):

| Eksperimen | backboard | ball | rim |
|---|---|---|---|
| E1_pretrained | «ISI» | «ISI» | «ISI» |
| E2_scratch | «ISI» | «ISI» | «ISI» |
| E3_tuning | «ISI» | «ISI» | «ISI» |

## 6. Hasil akhir

Model terpilih berdasarkan mAP50-95 val tertinggi: **«ISI, mis. E3_tuning»**.
Model ini diuji pada data test **satu kali saja**:

| mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|
| «ISI» | «ISI» | «ISI» | «ISI» |

Grafik dan confusion matrix: `hasil/TEST_«ISI»/`. Contoh prediksi: `hasil/prediksi_test/`.

## 7. Analisis

**E1 dibanding E2 (manfaat transfer learning).** «ISI: selisih mAP50-95 antara E1 dan E2,
dan penjelasan singkat. Dengan hanya 124 frame latih, bobot pretrained biasanya jauh
lebih baik daripada bobot acak. Tulis apakah hasilmu sesuai.»

**E1 dibanding E3 (pengaruh ukuran input).** «ISI: apakah imgsz 640 menaikkan mAP,
khususnya untuk `ball` yang kecil (sekitar 26×35 piksel pada frame 640×480), dan apa
konsekuensinya untuk kecepatan di Orange Pi 3B.»

**Kelas terlemah.** «ISI: lihat tabel AP per kelas dan confusion matrix. Sebutkan kelas
yang paling sering salah dan penyebab yang kamu amati, mis. bola tertukar dengan benda
kuning lain, atau ring tertutup.»

**Dibanding target.** «ISI: sebutkan metrik yang tercapai dan yang belum, serta langkah
perbaikan yang dipilih.»

## 8. Keterbatasan

- Dataset kecil (124 frame latih, 23 val, 26 test), sehingga selisih beberapa poin mAP
  bisa hanya derau dan belum tentu perbedaan nyata.
- Seluruh data berasal dari satu sesi, satu lokasi, satu kondisi cahaya, dan satu kamera
  (e-con), sedangkan robot akan memakai Logitech. Kinerja pada kamera, cahaya, atau
  latar lain belum terbukti.
- Hanya satu seed per eksperimen, jadi belum ada estimasi variasi antar-run.
- Anotasi hanya diperiksa pada tiga kotak yang diketahui salah.
- FPS dan latensi pada Orange Pi 3B «belum diukur / sudah diukur: ISI».

## 9. Rencana berikutnya

1. Tambah frame dari kamera Logitech dengan variasi jarak, sudut, dan cahaya.
2. Ekspor `best.pt` ke ONNX, lalu konversi ke RKNN (INT8) dan ukur FPS serta latensi
   langsung di Orange Pi 3B.
3. Ulangi eksperimen dengan beberapa seed dan data baru.

## 10. Struktur folder

```
Pj1_YOLO/
├── README.md
├── data.yaml
├── dataset_v1.zip
├── colab/Pj1_YOLO_Colab.ipynb
├── hasil/
│   ├── log_eksperimen.csv
│   ├── hasil_praktik_4222401028.txt
│   ├── E1_pretrained/
│   ├── E2_scratch/
│   ├── E3_tuning/
│   └── TEST_«ISI»/
└── model/best.pt
```

## 11. Cara menjalankan ulang

1. Upload `dataset_v1.zip` ke Google Drive di folder `MyDrive/pj1/`.
2. Buka `colab/Pj1_YOLO_Colab.ipynb` di Google Colab, pilih runtime GPU (T4).
3. Jalankan semua sel dari atas ke bawah. Hasil disimpan ke `MyDrive/pj1/runs/`.
