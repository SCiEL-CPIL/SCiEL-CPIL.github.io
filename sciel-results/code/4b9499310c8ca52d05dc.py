import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Step 1: Construct the 256-word linear code
    # This is the [17, 8, 6] code derived from the [18, 9, 6] extended QR code.
    
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
        
    # Initialize code
    code = np.array(words_short, dtype=np.int64)
    best_code = code.copy()
    best_size = len(code)
    
    # Precompute popcount table
    popcount_table = np.array([i.bit_count() for i in range(2**17)], dtype=np.int8)
    
    # Helper to check if a word is valid against the current code
    # Returns True if min distance >= 6
    def is_valid(w, code_arr):
        # w is int, code_arr is numpy array of ints
        diffs = code_arr ^ w
        dists = popcount_table[diffs]
        return np.min(dists) >= 6

    # Helper to check if a word is valid against code_arr excluding one index
    def is_valid_excluding(w, code_arr, exclude_idx):
        # Create a view or copy excluding the index
        # To avoid copy, we can just check all and ignore the one, but that's tricky with min
        # Simpler: just check all, if the one excluded is the only violation, it's valid.
        # But we need to be sure.
        # Let's just do the full check and handle the exclusion logic if needed, 
        # or better, just check against the list minus that one.
        # For efficiency, since N is small, we can just do:
        diffs = code_arr ^ w
        dists = popcount_table[diffs]
        # If the minimum distance is < 6, it must NOT be at the excluded index
        min_d = np.min(dists)
        if min_d >= 6:
            return True
        # If min_d < 6, check if the violation is ONLY at the excluded index
        # Find indices where dist < 6
        bad_indices = np.where(dists < 6)[0]
        if len(bad_indices) == 1 and bad_indices[0] == exclude_idx:
            return True
        return False

    rng = np.random.default_rng(seed=42)
    
    # Main search loop
    # Strategy:
    # 1. Try to add a word.
    # 2. If fail, try a swap.
    # 3. If swap succeeds, try to add again.
    
    no_add_count = 0
    max_no_add = 1000 # If we can't add for 1000 tries, force more swaps
    
    while time.time() < deadline - 2.0:
        current_size = len(code)
        
        # 1. Try to add a word
        added = False
        # Generate candidates
        # Mix of random and local
        for _ in range(50): # Try 50 candidates per "add attempt"
            if time.time() > deadline - 2.0:
                break
                
            # 50% random, 50% neighbor of random codeword
            if rng.random() < 0.5:
                w = rng.integers(0, 2**17)
            else:
                base_idx = rng.integers(0, current_size)
                base = code[base_idx]
                # Flip 1 to 3 bits
                num_flips = rng.integers(1, 4)
                w = base
                for _ in range(num_flips):
                    bit_pos = rng.integers(0, 17)
                    w ^= (1 << bit_pos)
                    
            if is_valid(w, code):
                code = np.append(code, w)
                added = True
                break
                
        if added:
            no_add_count = 0
            if current_size + 1 > best_size:
                best_size = current_size + 1
                best_code = code.copy()
        else:
            no_add_count += 1
            # 2. Try a swap
            # Pick a random word to remove
            idx_remove = rng.integers(0, current_size)
            removed_word = code[idx_remove]
            
            # Try to find a replacement
            swap_success = False
            for _ in range(20): # Try 20 candidates for swap
                if time.time() > deadline - 2.0:
                    break
                    
                # Generate candidate
                if rng.random() < 0.5:
                    w_new = rng.integers(0, 2**17)
                else:
                    # Neighbor of the removed word or a random other word
                    if rng.random() < 0.5:
                        base = removed_word
                    else:
                        base = code[rng.integers(0, current_size)]
                    num_flips = rng.integers(1, 4)
                    w_new = base
                    for _ in range(num_flips):
                        bit_pos = rng.integers(0, 17)
                        w_new ^= (1 << bit_pos)
                        
                # Check if w_new is valid against all except idx_remove
                if is_valid_excluding(w_new, code, idx_remove):
                    # Perform swap
                    # Remove idx_remove, add w_new
                    # Numpy delete is slow, but N is small.
                    # Or just mask
                    mask = np.ones(current_size, dtype=bool)
                    mask[idx_remove] = False
                    code = code[mask]
                    code = np.append(code, w_new)
                    swap_success = True
                    break
            
            # If no swap, we stay in the same configuration.
            # To avoid infinite loops of failing swaps, we might want to restart 
            # or just continue.
            
    return list(best_code)
