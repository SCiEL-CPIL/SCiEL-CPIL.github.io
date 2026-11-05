"""Construct largest 17-bit binary code with min Hamming distance 6."""
import numpy as np
import time

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


def _popcount17(x):
    """Fast popcount for 17-bit values using a precomputed table."""
    # Use a global table
    return _PC17[x]


def _compute_forbidden(code_words, N=17, max_dist=5):
    """Compute all words within max_dist of any code word."""
    code_arr = np.array(code_words, dtype=np.int64)
    n_code = len(code_arr)
    
    forbidden = np.zeros(2**N, dtype=np.bool_)
    
    # For each code word, mark all words within distance max_dist
    # This is expensive, so we use a smarter approach:
    # For a 17-bit word, the number of words within distance 5 is:
    # C(17,0)+C(17,1)+C(17,2)+C(17,3)+C(17,4)+C(17,5) = 1+17+136+680+2380+6188 = 9402
    # With 256 code words, that's potentially 256*9402 ≈ 2.4M operations (with overlaps)
    
    # Better: iterate over all 131072 candidates and check if any code word is within dist 5
    all_candidates = np.arange(2**N, dtype=np.int64)
    
    # Batch process: for each candidate, compute min distance to any code word
    # Process in batches to keep memory reasonable
    batch_size = 10000
    for start in range(0, 2**N, batch_size):
        end = min(start + batch_size, 2**N)
        batch = all_candidates[start:end]
        
        # Compute XOR of each candidate with each code word
        # Shape: (batch_size, n_code)
        diffs = batch[:, None] ^ code_arr[None, :]
        
        # Compute popcount for each diff
        # Use the precomputed table
        pc = _PC17[diffs]
        
        # Min distance across all code words
        min_dists = np.min(pc, axis=1)
        
        # Mark forbidden if min_dist <= max_dist
        forbidden[start:end] = (min_dists <= max_dist)
    
    return forbidden


def construct():
    """Construct the largest set of 17-bit words with pairwise Hamming distance at least 6."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 17
    MIN_DIST = 6
    NUM_WORDS = 2**N
    
    # Build popcount table for all 17-bit values
    global _PC17
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
    
    # Step 2: Compute forbidden set (words within distance 5 of any codeword)
    if time.time() > deadline - 40.0:
        return best_words
    
    forbidden = _compute_forbidden(best_words, N, max_dist=5)
    
    # Get list of candidates not in forbidden set and not already in code
    code_set = set(best_words)
    candidates = np.array([w for w in range(NUM_WORDS) if not forbidden[w] and w not in code_set], dtype=np.int64)
    
    if len(candidates) == 0:
        return best_words
    
    # Step 3: Greedy addition with multiple restarts
    # For each restart, try a different ordering of candidates
    
    current_list = list(best_words)
    current_set = set(current_list)
    current_arr = np.array(current_list, dtype=np.int64)
    
    # Try to add candidates greedily
    # Order candidates by their "conflict potential" - words that are far from each other
    # Simple heuristic: try in random order with different seeds
    
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(len(candidates))
    ordered_candidates = candidates[perm]
    
    added_count = 0
    for w in ordered_candidates:
        if time.time() > deadline - 10.0:
            break
        
        w = int(w)
        if w in current_set:
            continue
        
        # Check distance to all current words
        diffs = current_arr ^ w
        dists = _PC17[diffs]
        min_d = int(np.min(dists))
        
        if min_d >= MIN_DIST:
            current_list.append(w)
            current_set.add(w)
            current_arr = np.append(current_arr, w)
            added_count += 1
    
    if len(current_list) > best_size:
        best_size = len(current_list)
        best_words = current_list
    
    # Step 4: Local search - try to add more words by removing some existing ones
    # Try swap-based local search
    if time.time() > deadline - 5.0:
        return best_words
    
    # Try a few more random restarts with different orderings
    for restart in range(1, 20):
        if time.time() > deadline - 5.0:
            break
        
        rng = np.random.default_rng(seed=restart * 7919)
        perm = rng.permutation(len(candidates))
        ordered_candidates = candidates[perm]
        
        current_list = list(best_words)
        current_set = set(current_list)
        current_arr = np.array(current_list, dtype=np.int64)
        
        for w in ordered_candidates:
            if time.time() > deadline - 2.0:
                break
            
            w = int(w)
            if w in current_set:
                continue
            
            diffs = current_arr ^ w
            dists = _PC17[diffs]
            min_d = int(np.min(dists))
            
            if min_d >= MIN_DIST:
                current_list.append(w)
                current_set.add(w)
                current_arr = np.append(current_arr, w)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_words = current_list
    
    return best_words

# EVOLVE-BLOCK-END
