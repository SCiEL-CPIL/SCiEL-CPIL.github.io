import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 55.0  # Safety margin before 60s limit

    # --- Reed-Muller RM(1,4) seed (32 words, length 16, padded to 18) ---
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)

    word_list = words[:]
    word_set = set(words)

    # Convert to numpy bit matrix for fast batch distance checks
    codebook = np.zeros((len(words), N), dtype=np.uint8)
    for i, w in enumerate(words):
        for j in range(N):
            codebook[i, j] = (w >> j) & 1

    rng = np.random.default_rng(42)

    # --- Greedy random augmentation ---
    # Use batch generation + vectorized distance checks.
    # When the code grows large, reduce batch size to keep per-iteration cost manageable.
    
    # Precompute a lookup: for speed, we can use the bit matrix directly.
    
    while time.time() < deadline:
        # Adaptive batch size: smaller as code grows to avoid O(M*B) blowup
        M = len(word_list)
        # Target: each iteration should take < ~0.5s
        # distance check is O(M * batch) XOR ops; with M up to few thousand, keep batch moderate
        if M < 100:
            batch_size = 20000
        elif M < 500:
            batch_size = 5000
        elif M < 2000:
            batch_size = 1000
        else:
            batch_size = 200

        # Time-check before expensive work
        if time.time() > deadline:
            break

        candidates = rng.integers(0, 1 << N, size=batch_size)

        # Convert candidates to bit matrix
        cand_bits = np.zeros((batch_size, N), dtype=np.uint8)
        for c in range(batch_size):
            w = int(candidates[c])
            for j in range(N):
                cand_bits[c, j] = (w >> j) & 1

        # Skip candidates already in the set (distance 0)
        # Quick filter: check membership via integer set
        valid_indices = []
        for c in range(batch_size):
            w = int(candidates[c])
            if w not in word_set:
                valid_indices.append(c)

        if not valid_indices:
            continue

        valid_indices = np.array(valid_indices, dtype=np.int64)
        cand_sub = cand_bits[valid_indices]
        cand_ints = candidates[valid_indices]

        # Compute min distance from each candidate to all existing codewords
        # xors: (batch_valid, M, N) -- this can be large; do in chunks if needed
        chunk_size = 200
        n_valid = len(valid_indices)
        
        newly_added = 0
        for start in range(0, n_valid, chunk_size):
            if time.time() > deadline:
                break
            end = min(start + chunk_size, n_valid)
            chunk = cand_sub[start:end]
            chunk_ints = cand_ints[start:end]

            # XOR: (chunk_len, M, N)
            xors = codebook[None, :, :] ^ chunk[:, None, :]
            dists = np.sum(xors, axis=2)  # (chunk_len, M)
            min_dists = np.min(dists, axis=1)  # (chunk_len,)

            # Add those with min_dist >= MIN_DIST
            for i in range(len(chunk)):
                if min_dists[i] >= MIN_DIST:
                    w = int(chunk_ints[i])
                    word_list.append(w)
                    word_set.add(w)
                    # Append row to codebook
                    codebook = np.vstack([codebook, chunk[i:i+1]])
                    newly_added += 1

        # If no additions in this whole batch, we could reduce batch or change strategy
        # but for now just continue

    return word_list
