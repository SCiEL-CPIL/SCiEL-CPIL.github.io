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


def _generate_distance_k_words(N, k):
    """Generate all N-bit words with Hamming weight exactly k."""
    words = []
    for positions in combinations(range(N), k):
        val = 0
        for p in positions:
            val |= (1 << p)
        words.append(val)
    return np.array(words, dtype=np.int64)


def construct():
    """Construct the largest set of 17-bit words with pairwise Hamming distance at least 6."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 17
    MIN_DIST = 6
    NUM_WORDS = 2**N
    
    # Build popcount table for all 17-bit values
    _PC17 = np.array([i.bit_count() for i in range(NUM_WORDS)], dtype=np.int8)
    
    # Step 1: Construct the 256-word linear code
    try:
        words = _build_linear_code()
    except Exception:
        words = [0]
    
    if not words:
        return [0]
    
    best_words = list(words)
    best_size = len(best_words)
    code_arr = np.array(best_words, dtype=np.int64)
    code_set = set(best_words)
    
    # Step 2: Generate candidate pool - words at distance exactly 6 from zero
    # Since the code is linear, any word compatible with the code can be XOR'd
    # with a codeword to give another compatible word. So we just need to find
    # words at distance >= 6 from ALL codewords.
    # Start with words at weight exactly 6 (distance 6 from zero)
    candidates = _generate_distance_k_words(N, 6)
    
    # Also add words at weight 7 and 8 that might be compatible
    # But focus first on weight 6 for efficiency
    # Actually, let's be more comprehensive: add weight 6, 7, 8, 9 candidates
    # that are not already in the code
    all_candidates_list = []
    for k in [6, 7, 8, 9, 10, 11]:
        if time.time() > deadline - 30.0:
            break
        cw = _generate_distance_k_words(N, k)
        # Filter out words already in code
        mask = np.array([w not in code_set for w in cw], dtype=bool)
        all_candidates_list.append(cw[mask])
    
    if all_candidates_list:
        all_candidates = np.concatenate(all_candidates_list)
    else:
        all_candidates = np.array([], dtype=np.int64)
    
    if len(all_candidates) == 0:
        return best_words
    
    # Step 3: Greedy addition with vectorized distance checks
    # Process candidates in batches, checking distance to all current codewords
    
    def try_add_candidates(cand_arr, current_list, current_set, current_arr, deadline):
        """Try to add candidates to the code using vectorized checks."""
        added = []
        n_cand = len(cand_arr)
        
        if n_cand == 0:
            return added, current_list, current_set, current_arr
        
        # Process in batches for memory efficiency
        batch_size = 2000
        for start in range(0, n_cand, batch_size):
            if time.time() > deadline - 5.0:
                break
            
            end = min(start + batch_size, n_cand)
            batch = cand_arr[start:end]
            
            # Compute distances from each candidate to each current codeword
            # Shape: (batch_size, len(current_arr))
            diffs = batch[:, None] ^ current_arr[None, :]
            dists = _PC17[diffs]
            
            # For each candidate, find min distance to any codeword
            min_dists = np.min(dists, axis=1)
            
            # Candidates with min_dist >= MIN_DIST are valid
            valid_mask = min_dists >= MIN_DIST
            valid_cands = batch[valid_mask]
            
            # Add valid candidates one by one (need to update current_arr)
            for w in valid_cands:
                if time.time() > deadline - 3.0:
                    break
                
                w = int(w)
                if w in current_set:
                    continue
                
                current_list.append(w)
                current_set.add(w)
                current_arr = np.append(current_arr, w)
                added.append(w)
        
        return added, current_list, current_set, current_arr
    
    # Try multiple random orderings to maximize additions
    rng = np.random.default_rng(seed=42)
    
    best_added = 0
    best_final_list = best_words
    
    for restart in range(30):
        if time.time() > deadline - 5.0:
            break
        
        perm = rng.permutation(len(all_candidates))
        ordered_cands = all_candidates[perm]
        
        current_list = list(best_words)
        current_set = set(current_list)
        current_arr = np.array(current_list, dtype=np.int64)
        
        added, current_list, current_set, current_arr = try_add_candidates(
            ordered_cands, current_list, current_set, current_arr, deadline
        )
        
        if len(added) > best_added:
            best_added = len(added)
            best_final_list = current_list
    
    if len(best_final_list) > best_size:
        best_size = len(best_final_list)
        best_words = best_final_list
    
    return best_words

# EVOLVE-BLOCK-END
