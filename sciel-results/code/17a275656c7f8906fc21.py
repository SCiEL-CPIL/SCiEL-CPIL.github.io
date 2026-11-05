import math
import time
import numpy as np

def construct():
    start_time = time.time()
    LIMIT = 115.0
    MAX_S = 10**12
    
    # 1. Generate primes up to 10^6
    limit_p = 10**6
    is_prime = np.ones(limit_p + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # 2. Efficient phi calculation for a range using NumPy
    def compute_phi_range(L, R):
        n = R - L + 1
        if n <= 0:
            return np.array([], dtype=np.int64)
        
        vals = np.arange(L, R + 1, dtype=np.int64)
        phis = np.arange(L, R + 1, dtype=np.int64)
        
        for p in primes:
            if p * p > R:
                break
            first = ((L + p - 1) // p) * p
            if first > R:
                continue
            start_idx = first - L
            indices = np.arange(start_idx, n, p)
            vals[indices] //= p
            phis[indices] -= phis[indices] // p
            
        mask = vals > 1
        if np.any(mask):
            phis[mask] -= phis[mask] // vals[mask]
            
        return phis

    # 3. Find longest run of distinct values
    def find_longest_run(phis):
        # phis is numpy array
        # Convert to list for sliding window? Or use pandas?
        # Simple Python loop is fine for N=100000
        # But N=1000000 might be slow in pure Python.
        # 10^6 iterations in Python ~ 0.05s. Acceptable.
        
        seen = {}
        l = 0
        max_len = 0
        max_start_idx = 0
        
        # phis is numpy array, iterate over it
        for r, val in enumerate(phis):
            val_int = int(val)
            if val_int in seen and seen[val_int] >= l:
                l = seen[val_int] + 1
            else:
                if r - l + 1 > max_len:
                    max_len = r - l + 1
                    max_start_idx = l
            seen[val_int] = r
            
        return max_len, max_start_idx

    best_k = 1
    best_s = 1
    
    B = 1000000
    OVERLAP = 200
    
    current_end = MAX_S
    
    while current_end > 1:
        if time.time() - start_time > LIMIT:
            break
            
        block_start = max(1, current_end - B + 1)
        
        phis = compute_phi_range(block_start, current_end)
        
        len_in_block, idx_in_block = find_longest_run(phis)
        
        if len_in_block > best_k:
            best_k = len_in_block
            best_s = block_start + idx_in_block
            
            if best_k >= 200:
                return {'start': best_s, 'length': 200}
        
        if block_start == 1:
            break
            
        next_end = block_start - 1 + OVERLAP
        if next_end >= current_end:
            next_end = current_end - 1
        current_end = next_end

    return {'start': best_s, 'length': best_k}
