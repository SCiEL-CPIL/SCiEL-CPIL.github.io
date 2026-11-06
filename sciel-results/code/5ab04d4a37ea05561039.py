"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    L = 1001
    R = 1000024
    N = R - L + 1
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Compute smooth parts using vectorized approach
    # For each number, divide out all factors >= 23
    # Equivalently: smooth_part = number / (product of prime powers >= 23)
    # But easier: for each prime < 24, multiply the power into smooth_part
    
    # Use an array of floats for speed, then convert
    smooth = np.ones(N, dtype=np.int64)
    
    for p in primes:
        # Find indices of multiples of p in [L, R]
        # m = L + idx, so idx = m - L
        # m ranges from first multiple of p >= L to last <= R, step p
        start_m = ((L + p - 1) // p) * p
        # idx for start_m
        start_idx = start_m - L
        # step in idx is p
        for m in range(start_m, R + 1, p):
            idx = m - L
            # Divide m by p repeatedly, multiplying smooth[idx] by p each time
            x = m
            while x % p == 0:
                smooth[idx] *= p
                x //= p
    
    if time.time() > deadline:
        return 1000
    
    # Convert to list for faster sliding window (numpy indexing is slower for random access)
    sp = smooth.tolist()
    
    max_start_idx = 1000000 + 1 - L  # 999000
    
    best_n = 1000
    min_distinct = float('inf')
    
    # Use a dict for frequency counting
    freq = {}
    for j in range(24):
        v = sp[j]
        freq[v] = freq.get(v, 0) + 1
    
    current_distinct = len(freq)
    min_distinct = current_distinct
    best_n = 1000
    
    if min_distinct == 1:
        return best_n
    
    for start_idx in range(1, max_start_idx + 1):
        old_val = sp[start_idx - 1]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        new_val = sp[start_idx + 23]
        if new_val not in freq:
            current_distinct += 1
            freq[new_val] = 1
        else:
            freq[new_val] += 1
        
        if current_distinct < min_distinct:
            min_distinct = current_distinct
            best_n = start_idx + L - 1
            if min_distinct == 1:
                break
        
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
