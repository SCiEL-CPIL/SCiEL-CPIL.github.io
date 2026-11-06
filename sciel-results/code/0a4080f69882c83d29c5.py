"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_N = 1000000
    MIN_N = 1000
    WINDOW_SIZE = 24
    
    # We need small parts for numbers up to MAX_N + WINDOW_SIZE
    M = MAX_N + WINDOW_SIZE + 10
    
    # Precompute small parts using NumPy for speed
    # sp[x] = product of prime factors < 24 of x
    sp = np.ones(M + 1, dtype=np.int64)
    
    for p in primes:
        # For each multiple of p, compute the p-part
        multiples = np.arange(p, M + 1, p)
        # For each multiple, count how many times p divides it
        # Use a loop but vectorize where possible
        for m in multiples:
            temp = m
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            sp[m] *= p ** cnt
    
    # Map smooth part values to compact indices
    unique_vals = {}
    sp_idx = np.zeros(M + 1, dtype=np.int32)
    for i in range(M + 1):
        v = int(sp[i])
        if v not in unique_vals:
            unique_vals[v] = len(unique_vals)
        sp_idx[i] = unique_vals[v]
    
    K = len(unique_vals)
    
    # Sliding window with frequency counter
    counts = np.zeros(K, dtype=np.int32)
    distinct = 0
    
    # Initialize counts for the first window: sp[MIN_N+1] .. sp[MIN_N+24]
    for i in range(1, WINDOW_SIZE + 1):
        idx = int(sp_idx[MIN_N + i])
        if counts[idx] == 0:
            distinct += 1
        counts[idx] += 1
        
    best_n = MIN_N
    best_distinct = distinct
    
    # Slide the window
    for n in range(MIN_N, MAX_N + 1):
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            if best_distinct <= 2:  # Very unlikely to go much lower
                break
        
        # Check time every 5000 iterations
        if (n - MIN_N) % 5000 == 0:
            if time.time() > deadline:
                break
        
        if n < MAX_N:
            # Remove sp[n+1]
            idx_out = int(sp_idx[n + 1])
            counts[idx_out] -= 1
            if counts[idx_out] == 0:
                distinct -= 1
            # Add sp[n+1+WINDOW_SIZE]
            idx_in = int(sp_idx[n + 1 + WINDOW_SIZE])
            if counts[idx_in] == 0:
                distinct += 1
            counts[idx_in] += 1
            
    return int(best_n)
# EVOLVE-BLOCK-END
