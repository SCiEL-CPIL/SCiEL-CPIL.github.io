"""Construct largest 17-bit binary code with min Hamming distance 6."""
import numpy as np
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_linear_code():
    """Construct the 256-word [17,8,6] code from the [18,9,6] extended QR code."""
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
    
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
    
    words_18 = []
    for w in words_17:
        parity = int(np.sum(w) % 2)
        word_ext = np.append(w, parity)
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
    
    return words_short


def _enumerate_dist_k_words(center, k, n):
    """Enumerate all n-bit words at Hamming distance exactly k from center."""
    all_positions = list(range(n))
    results = []
    for subset in combinations(all_positions, k):
        mask = 0
        for pos in subset:
            mask |= (1 << pos)
        results.append(center ^ mask)
    return results


def construct():
    """Construct the largest set of 17-bit words with pairwise Hamming distance at least 6.
    
    Uses focused candidate enumeration: words at Hamming distance exactly 6 from
    multiple codewords in the linear code, with vectorized batched compatibility checks.
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    # Step 1: Construct the 256-word linear code
    try:
        words = _build_linear_code()
    except Exception:
        words = [0]
    
    if not words:
        return [0]
    
    best_words = list(words)
    best_size = len(best_words)
    
    # Precompute popcount table for all 17-bit values
    popcount_table = np.array([i.bit_count() for i in range(2**N)], dtype=np.int8)
    
    # Step 2: Build expanded candidate pool
    # Enumerate words at distances 5-8 from multiple codewords
    candidate_set = set()
    
    # Use more diverse anchors including 0 and spread-out codewords
    anchors = [0]
    rng_anchor = np.random.default_rng(seed=12345)
    for idx in rng_anchor.choice(len(words), size=min(20, len(words)), replace=False):
        anchors.append(words[idx])
    
    # Generate candidates at multiple distances from each anchor
    for anchor in anchors:
        for dist in [5, 6, 7, 8]:
            dist_words = _enumerate_dist_k_words(anchor, dist, N)
            candidate_set.update(dist_words)
    
    # Remove words already in the code
    code_set = set(best_words)
    candidate_list = [w for w in candidate_set if w not in code_set]
    
    # Shuffle candidates for diversity
    rng = np.random.default_rng(seed=42)
    rng.shuffle(candidate_list)
    
    # Step 3: Batched greedy addition
    current_list = list(best_words)
    current_set = set(current_list)
    current_arr = np.array(current_list, dtype=np.int64)
    
    # Process candidates in batches for vectorized efficiency
    batch_size = 1000
    added_this_pass = 0
    
    for pass_num in range(10):  # More passes with larger candidate pool
        if time.time() > deadline - 5.0:
            break
        
        added_this_pass = 0
        remaining = []
        
        for i in range(0, len(candidate_list), batch_size):
            if time.time() > deadline - 3.0:
                break
            
            batch = candidate_list[i:i+batch_size]
            batch_arr = np.array(batch, dtype=np.int64)
            
            # Vectorized: compute distances from each batch word to all current codewords
            diffs = batch_arr[:, None] ^ current_arr[None, :]
            dists = popcount_table[diffs]
            
            # For each batch word, check if min distance to current set >= MIN_DIST
            min_dists = np.min(dists, axis=1)
            valid_mask = min_dists >= MIN_DIST
            
            # Add valid words one by one (to update current set incrementally)
            for j in range(len(batch)):
                if valid_mask[j]:
                    w = int(batch_arr[j])
                    if w not in current_set:
                        current_list.append(w)
                        current_set.add(w)
                        current_arr = np.append(current_arr, w)
                        added_this_pass += 1
                    valid_mask[j] = False
                else:
                    remaining.append(int(batch_arr[j]))
        
        # Keep only candidates that weren't added for next pass
        if added_this_pass == 0:
            break
        
        # For next pass, keep candidates that are still not in the set
        candidate_list = [w for w in candidate_list if w not in current_set]
        
        # Shuffle again for next pass
        rng.shuffle(candidate_list)
        
        # Update best
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_words = current_list
    
    # Step 4: Additional random candidates from full space if time remains
    if time.time() < deadline - 10.0:
        # Add more random candidates beyond the focused pool
        rng2 = np.random.default_rng(seed=99999)
        n_extra = min(100000, 2**N - len(current_set))
        extra_candidates = rng2.integers(0, 2**N, size=n_extra).tolist()
        
        extra_candidates = [w for w in extra_candidates if w not in current_set]
        rng2.shuffle(extra_candidates)
        
        for i in range(0, len(extra_candidates), batch_size):
            if time.time() > deadline - 3.0:
                break
            
            batch = extra_candidates[i:i+batch_size]
            batch_arr = np.array(batch, dtype=np.int64)
            diffs = batch_arr[:, None] ^ current_arr[None, :]
            dists = popcount_table[diffs]
            min_dists = np.min(dists, axis=1)
            
            for j in range(len(batch)):
                if min_dists[j] >= MIN_DIST:
                    w = int(batch_arr[j])
                    if w not in current_set:
                        current_list.append(w)
                        current_set.add(w)
                        current_arr = np.append(current_arr, w)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_words = current_list
    
    return best_words

# EVOLVE-BLOCK-END
