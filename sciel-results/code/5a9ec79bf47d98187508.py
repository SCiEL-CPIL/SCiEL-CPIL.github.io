import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Build the [18, 6, 6] extended QR(17) code as a strong seed
    row0_positions = [0, 1, 3, 7, 8, 12, 14, 15]
    G = []
    for i in range(6):
        row = 0
        for pos in row0_positions:
            shifted = (pos + i) % 17
            row |= (1 << shifted)
        G.append(row)

    qr_words = []
    for mask in range(64):
        word = 0
        for i in range(6):
            if (mask >> i) & 1:
                word ^= G[i]
        parity = word.bit_count() % 2
        extended = word | (parity << 17)
        qr_words.append(extended)

    current_words = list(qr_words)
    current_set = set(current_words)
    current_np = np.array([list(map(int, format(w, '018b'))) for w in current_words], dtype=np.uint8)
    best_size = len(current_words)
    best_words = list(current_words)

    def _int_to_np(w):
        return np.array(list(map(int, format(w, '018b'))), dtype=np.uint8)

    def _np_to_int(arr):
        val = 0
        for b in arr:
            val = (val << 1) | int(b)
        return val

    # Vectorized batched greedy search
    BATCH_SIZE = 4096
    all_indices = np.arange(1 << N)
    rng = np.random.default_rng(42)
    rng.shuffle(all_indices)

    for start_idx in range(0, len(all_indices), BATCH_SIZE):
        if time.time() > deadline - 10.0:
            break
        end_idx = min(start_idx + BATCH_SIZE, len(all_indices))
        batch = all_indices[start_idx:end_idx]

        # Convert batch to (B, N) binary array
        batch_np = np.array([list(map(int, format(w, '018b'))) for w in batch], dtype=np.uint8)

        # Vectorized distance check
        # batch: (B, N), current: (M, N)
        xor_res = batch_np[:, np.newaxis, :] ^ current_np[np.newaxis, :, :]
        dists = np.sum(xor_res, axis=2)
        min_dists = np.min(dists, axis=1)
        valid_mask = min_dists >= MIN_DIST

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            for vi in valid_indices:
                candidate_int = int(batch[vi])
                if candidate_int not in current_set:
                    current_words.append(candidate_int)
                    current_set.add(candidate_int)
                    current_np = np.vstack([current_np, _int_to_np(candidate_int)])
                    if len(current_words) > best_size:
                        best_size = len(current_words)
                        best_words = list(current_words)

    # Local search: 1-bit flips
    if time.time() < deadline - 8.0:
        flipped_candidates = set()
        for w in current_words:
            for bit in range(N):
                flipped = w ^ (1 << bit)
                if flipped not in current_set:
                    flipped_candidates.add(flipped)

        flipped_list = list(flipped_candidates)
        if flipped_list:
            flipped_np = np.array([list(map(int, format(w, '018b'))) for w in flipped_list], dtype=np.uint8)
            xor_res = flipped_np[:, np.newaxis, :] ^ current_np[np.newaxis, :, :]
            dists = np.sum(xor_res, axis=2)
            min_dists = np.min(dists, axis=1)
            valid_mask = min_dists >= MIN_DIST
            valid_indices = np.where(valid_mask)[0]
            for vi in valid_indices:
                candidate_int = int(flipped_list[vi])
                if candidate_int not in current_set:
                    current_words.append(candidate_int)
                    current_set.add(candidate_int)
                    current_np = np.vstack([current_np, _int_to_np(candidate_int)])
                    if len(current_words) > best_size:
                        best_size = len(current_words)
                        best_words = list(current_words)

    # Local search: 2-bit flips
    if time.time() < deadline - 5.0:
        two_flip_candidates = set()
        for w in current_words:
            for b1 in range(N):
                for b2 in range(b1 + 1, N):
                    flipped = w ^ (1 << b1) ^ (1 << b2)
                    if flipped not in current_set:
                        two_flip_candidates.add(flipped)

        two_flip_list = list(two_flip_candidates)
        # Process in batches to avoid memory issues
        TF_BATCH = 2048
        for tf_start in range(0, len(two_flip_list), TF_BATCH):
            if time.time() > deadline - 2.0:
                break
            tf_end = min(tf_start + TF_BATCH, len(two_flip_list))
            tf_batch = two_flip_list[tf_start:tf_end]
            tf_np = np.array([list(map(int, format(w, '018b'))) for w in tf_batch], dtype=np.uint8)
            xor_res = tf_np[:, np.newaxis, :] ^ current_np[np.newaxis, :, :]
            dists = np.sum(xor_res, axis=2)
            min_dists = np.min(dists, axis=1)
            valid_mask = min_dists >= MIN_DIST
            valid_indices = np.where(valid_mask)[0]
            for vi in valid_indices:
                candidate_int = int(tf_batch[vi])
                if candidate_int not in current_set:
                    current_words.append(candidate_int)
                    current_set.add(candidate_int)
                    current_np = np.vstack([current_np, _int_to_np(candidate_int)])
                    if len(current_words) > best_size:
                        best_size = len(current_words)
                        best_words = list(current_words)

    return best_words
