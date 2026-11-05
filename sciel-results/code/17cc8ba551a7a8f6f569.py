"""Construct largest 17-bit binary code with min Hamming distance 6."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_PC17 = None

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
    
    return words_short


def _meet_in_middle_verify(code_arr, N=17, min_dist=6):
    """
    Use meet-in-the-middle to verify whether any word outside the code
    is at distance >= min_dist from ALL codewords.
    
    Split 17 bits into low 8 bits and high 9 bits.
    For each candidate, compute min distance using precomputed tables.
    Returns True if code is maximal (no word can be added).
    """
    global _PC17
    n_code = len(code_arr)
    
    # Split codewords into low/high parts
    low_parts = (code_arr & 0xFF).astype(np.int64)       # 8 bits
    high_parts = (code_arr >> 8).astype(np.int64)         # 9 bits
    
    # Precompute popcount for 8-bit and 9-bit values
    pc8 = np.array([i.bit_count() for i in range(256)], dtype=np.int8)
    pc9 = np.array([i.bit_count() for i in range(512)], dtype=np.int8)
    
    # For each of the 2^17 candidates, check if min dist >= min_dist
    # Process in batches
    batch_size = 8192
    all_candidates = np.arange(2**N, dtype=np.int64)
    
    for start in range(0, 2**N, batch_size):
        end = min(start + batch_size, 2**N)
        batch = all_candidates[start:end]
        bl = (batch & 0xFF).astype(np.int64)
        bh = (batch >> 8).astype(np.int64)
        
        # Compute distances to all codewords using split tables
        # d(c, w) = pc8(c_low ^ w_low) + pc9(c_high ^ w_high)
        # Shape: (batch, n_code)
        d_low = pc8[bl[:, None] ^ low_parts[None, :]]
        d_high = pc9[bh[:, None] ^ high_parts[None, :]]
        total_dists = d_low + d_high
        min_d = np.min(total_dists, axis=1)
        
        # If any candidate has min_d >= min_dist, code is NOT maximal
        if np.any(min_d >= min_dist):
            return False
    
    return True


def _swap_local_search(code_list, pc_table, N=17, min_dist=6, deadline=None):
    """
    Try swap-based local search: remove one codeword, try to add two new ones.
    This can potentially grow the code beyond the linear code size.
    """
    n = len(code_list)
    code_arr = np.array(code_list, dtype=np.int64)
    code_set = set(code_list)
    
    # Precompute all-pairs distances within the code
    # For a code of size 256, this is 256*256 = 65536 entries - manageable
    # But we'll compute on the fly to save memory
    
    best_list = list(code_list)
    best_size = n
    
    # Try random swaps
    rng = np.random.default_rng(seed=12345)
    
    # Precompute popcount for 8 and 9 bit halves for faster distance
    pc8 = np.array([i.bit_count() for i in range(256)], dtype=np.int8)
    pc9 = np.array([i.bit_count() for i in range(512)], dtype=np.int8)
    
    low_parts = (code_arr & 0xFF).astype(np.int64)
    high_parts = (code_arr >> 8).astype(np.int64)
    
    num_attempts = 5000
    for attempt in range(num_attempts):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        # Pick a random codeword to potentially remove
        idx_remove = rng.integers(0, len(best_list))
        removed_word = best_list[idx_remove]
        
        # Try to find two words that can replace it
        # Sample random candidates
        found_replacements = []
        remaining = [w for i, w in enumerate(best_list) if i != idx_remove]
        remaining_arr = np.array(remaining, dtype=np.int64)
        remaining_low = (remaining_arr & 0xFF).astype(np.int64)
        remaining_high = (remaining_arr >> 8).astype(np.int64)
        
        # Try up to 50 random candidates for first replacement
        for _ in range(50):
            if deadline is not None and time.time() > deadline - 3.0:
                break
            cand1 = int(rng.integers(0, 2**N))
            if cand1 in code_set and cand1 != removed_word:
                continue
            
            # Check cand1 against remaining
            c1_low = cand1 & 0xFF
            c1_high = cand1 >> 8
            d1 = pc8[c1_low ^ remaining_low] + pc9[c1_high ^ remaining_high]
            if np.min(d1) < min_dist:
                continue
            
            # Check cand1 against removed_word (should be fine since we're replacing)
            # Actually we need cand1 to be far from all remaining, which we checked
            
            # Now try to find cand2 that's far from remaining + cand1
            for _ in range(30):
                if deadline is not None and time.time() > deadline - 3.0:
                    break
                cand2 = int(rng.integers(0, 2**N))
                if cand2 in code_set and cand2 != removed_word:
                    continue
                if cand2 == cand1:
                    continue
                
                # Check cand2 vs cand1
                d12 = pc_table[cand1 ^ cand2]
                if d12 < min_dist:
                    continue
                
                # Check cand2 vs remaining
                c2_low = cand2 & 0xFF
                c2_high = cand2 >> 8
                d2 = pc8[c2_low ^ remaining_low] + pc9[c2_high ^ remaining_high]
                if np.min(d2) < min_dist:
                    continue
                
                # Both found! Create new code
                new_list = remaining + [cand1, cand2]
                if len(new_list) > best_size:
                    best_size = len(new_list)
                    best_list = new_list
                    code_set = set(best_list)
                    code_arr = np.array(best_list, dtype=np.int64)
                    low_parts = (code_arr & 0xFF).astype(np.int64)
                    high_parts = (code_arr >> 8).astype(np.int64)
                break
            if len(best_list) > n:
                break
    
    return best_list, best_size


def construct():
    """Construct the largest set of 17-bit words with pairwise Hamming distance at least 6."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    N = 17
    MIN_DIST = 6
    NUM_WORDS = 2**N
    
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
    
    # Step 2: Meet-in-the-middle verification of maximality
    if time.time() < deadline - 30.0:
        code_arr = np.array(best_words, dtype=np.int64)
        is_maximal = _meet_in_middle_verify(code_arr, N, MIN_DIST)
        # If maximal, we know no single word can be added
        # Focus on swap-based search
        if is_maximal:
            # Code is maximal - try swap-based local search
            if time.time() < deadline - 5.0:
                new_list, new_size = _swap_local_search(
                    best_words, _PC17, N, MIN_DIST, deadline
                )
                if new_size > best_size:
                    best_size = new_size
                    best_words = new_list
        else:
            # Not maximal - there are words that can be added
            # Use greedy addition
            current_list = list(best_words)
            current_set = set(current_list)
            current_arr = np.array(current_list, dtype=np.int64)
            
            rng = np.random.default_rng(seed=42)
            all_candidates = np.arange(NUM_WORDS, dtype=np.int64)
            perm = rng.permutation(len(all_candidates))
            ordered_candidates = all_candidates[perm]
            
            for w in ordered_candidates:
                if time.time() > deadline - 5.0:
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
    
    # Step 3: If we still have time, try more swap-based search
    if time.time() < deadline - 5.0 and best_size <= 256:
        new_list, new_size = _swap_local_search(
            best_words, _PC17, N, MIN_DIST, deadline
        )
        if new_size > best_size:
            best_size = new_size
            best_words = new_list
    
    return best_words

# EVOLVE-BLOCK-END
