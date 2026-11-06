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

def _enumerate_dist_k_words(N, k):
    """Enumerate all N-bit words with exactly k ones."""
    words = []
    for positions in combinations(range(N), k):
        val = 0
        for p in positions:
            val |= (1 << p)
        words.append(val)
    return words

def _try_add_words(base_words, candidates, popcount_table, deadline, min_dist=6):
    """Try to add candidates to base_words greedily. Returns list of added words."""
    current_arr = np.array(base_words, dtype=np.int64)
    current_set = set(base_words)
    added = []
    
    for w in candidates:
        if time.time() > deadline:
            break
        
        if w in current_set:
            continue
        
        diffs = current_arr ^ w
        dists = popcount_table[diffs]
        min_d = int(np.min(dists))
        
        if min_d >= min_dist:
            current_arr = np.append(current_arr, w)
            current_set.add(int(w))
            added.append(int(w))
    
    return added, current_arr, current_set

def construct():
    """Construct the largest set of 17-bit words with pairwise Hamming distance at least 6."""
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
    
    # Precompute popcount table
    popcount_table = np.array([i.bit_count() for i in range(2**N)], dtype=np.int8)
    
    # Step 2: Generate focused candidate pool - weight 6 words first (most likely to fit)
    candidates_w6 = _enumerate_dist_k_words(N, 6)
    candidates_w6_arr = np.array(candidates_w6, dtype=np.int64)
    
    code_set = set(best_words)
    mask = np.array([w not in code_set for w in candidates_w6_arr])
    candidates_w6_arr = candidates_w6_arr[mask]
    
    # Try adding weight-6 candidates
    added, _, _ = _try_add_words(best_words, candidates_w6_arr, popcount_table, deadline - 10.0)
    
    if len(added) > 0:
        current_list = best_words + added
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_words = current_list
    
    # Step 3: Try weight-7 candidates if time permits
    if time.time() < deadline - 15.0:
        candidates_w7 = _enumerate_dist_k_words(N, 7)
        candidates_w7_arr = np.array(candidates_w7, dtype=np.int64)
        code_set = set(best_words)
        mask = np.array([w not in code_set for w in candidates_w7_arr])
        candidates_w7_arr = candidates_w7_arr[mask]
        
        added, _, _ = _try_add_words(best_words, candidates_w7_arr, popcount_table, deadline - 5.0)
        
        if len(added) > 0:
            current_list = best_words + added
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_words = current_list
    
    # Step 4: Random restarts with shuffled candidates
    if time.time() < deadline - 10.0:
        all_candidates = np.concatenate([candidates_w6_arr, candidates_w7_arr])
        
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            rng = np.random.default_rng(seed=restart * 7919 + 12345)
            shuffled = all_candidates[rng.permutation(len(all_candidates))]
            
            current_arr = np.array(best_words, dtype=np.int64)
            current_set = set(best_words)
            current_list = list(best_words)
            
            for w in shuffled:
                if time.time() > deadline - 2.0:
                    break
                
                if w in current_set:
                    continue
                
                diffs = current_arr ^ w
                dists = popcount_table[diffs]
                min_d = int(np.min(dists))
                
                if min_d >= MIN_DIST:
                    current_list.append(int(w))
                    current_set.add(int(w))
                    current_arr = np.append(current_arr, w)
            
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_words = current_list
    
    return best_words

# EVOLVE-BLOCK-END
