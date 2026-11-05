import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Step 1: Construct the 256-word linear code [17,8,6]
    poly = 0x1D

    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break

    if alpha is None:
        return [0]

    Q = [1, 2, 4, 8, 9, 13, 15, 16]

    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g

    G = np.zeros((9, 17), dtype=np.int64)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1

    words_18 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        parity = int(np.sum(word) % 2)
        word_ext = np.append(word, parity)
        words_18.append(word_ext)

    words_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_short.append(val)

    if not words_short:
        return [0]

    # Precompute popcount table
    popcount_table = np.array([i.bit_count() for i in range(2**17)], dtype=np.int8)

    # Base code as numpy array
    base_code = np.array(words_short, dtype=np.int64)
    best_code = base_code.copy()
    best_size = len(base_code)

    # All possible 17-bit words
    all_words = np.arange(2**17, dtype=np.int64)

    # Create a set of indices in the base code for fast lookup
    base_set = set(int(x) for x in base_code)

    # Step 2: Vectorized greedy extension
    # Strategy: process candidates in shuffled order, in batches.
    # For each batch, compute distances to all current codewords vectorized.
    
    num_restarts = 30
    batch_size = 20000

    for restart in range(num_restarts):
        if time.time() > deadline - 3.0:
            break

        rng = np.random.default_rng(seed=restart * 12345 + 67890)
        candidates = rng.permutation(all_words)

        current_list = list(base_code)
        current_array = base_code.copy()
        current_set = set(current_list)

        # We'll track which candidates have been "used" (added to code)
        # Process in batches
        for batch_start in range(0, len(candidates), batch_size):
            if time.time() > deadline - 2.0:
                break

            batch_end = min(batch_start + batch_size, len(candidates))
            batch = candidates[batch_start:batch_end]

            # Filter out words already in the code
            mask = np.array([w not in current_set for w in batch], dtype=bool)
            if not np.any(mask):
                continue
            batch = batch[mask]

            if len(batch) == 0:
                continue

            # Compute distances from each candidate to all current codewords
            # batch shape: (B,), current_array shape: (C,)
            # We want dists[i, j] = popcount(batch[i] ^ current_array[j])
            # This is B x C matrix. For B=20000, C~258, that's ~5M entries - feasible.
            
            # Use broadcasting: batch[:, None] ^ current_array[None, :]
            # But this creates a B x C array which for 20000 x 258 = 5.16M int64 = 41MB
            # Then index into popcount_table.
            
            # To save memory, process in sub-batches if needed
            sub_batch_size = 5000
            for sb_start in range(0, len(batch), sub_batch_size):
                if time.time() > deadline - 2.0:
                    break
                sb_end = min(sb_start + sub_batch_size, len(batch))
                sub_batch = batch[sb_start:sb_end]
                
                # Compute distance matrix: (S, C)
                diffs = sub_batch[:, None] ^ current_array[None, :]
                dists = popcount_table[diffs]
                min_dists = np.min(dists, axis=1)
                
                # Find candidates with min distance >= 6
                valid_mask = min_dists >= 6
                if np.any(valid_mask):
                    valid_words = sub_batch[valid_mask]
                    for w in valid_words:
                        current_list.append(int(w))
                        current_set.add(int(w))
                    current_array = np.append(current_array, valid_words)

        if len(current_list) > best_size:
            best_size = len(current_list)
            best_code = np.array(current_list, dtype=np.int64)

    # Step 3: Local search / hill climbing to try to add more words
    # Try swapping: remove one word, add one or more
    if time.time() < deadline - 5.0:
        for restart in range(10):
            if time.time() > deadline - 3.0:
                break
            
            rng = np.random.default_rng(seed=restart * 99999 + 1)
            code_list = list(best_code)
            code_set = set(code_list)
            code_array = np.array(code_list, dtype=np.int64)
            
            # Try to add words that were previously rejected
            # Or try swap: remove a random word, see if we can add 2+ new ones
            for attempt in range(20):
                if time.time() > deadline - 3.0:
                    break
                
                # Pick a random word to potentially remove
                if len(code_list) <= 1:
                    break
                idx = rng.integers(0, len(code_list))
                removed = code_list[idx]
                
                # Try to add words that are distance >= 6 from all except removed
                # This is complex; instead just try adding from remaining candidates
                # Simple approach: try to add any word not in code that's >= 6 from all
                remaining = [w for w in all_words if w not in code_set]
                if not remaining:
                    break
                
                # Sample some candidates
                sample_size = min(5000, len(remaining))
                sample = rng.choice(remaining, size=sample_size, replace=False)
                
                added_any = False
                for w in sample:
                    if time.time() > deadline - 3.0:
                        break
                    # Check distance to all current codewords (including removed one temporarily)
                    diffs = code_array ^ w
                    dists = popcount_table[diffs]
                    if np.min(dists) >= 6:
                        code_list.append(int(w))
                        code_set.add(int(w))
                        code_array = np.append(code_array, w)
                        added_any = True
                
                if added_any and len(code_list) > best_size:
                    best_size = len(code_list)
                    best_code = np.array(code_list, dtype=np.int64)
                else:
                    # Restore
                    code_list = list(best_code)
                    code_set = set(code_list)
                    code_array = np.array(code_list, dtype=np.int64)

    return list(best_code)
