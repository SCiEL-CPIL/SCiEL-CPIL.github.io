"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy: Use an optimized LPF sieve with early termination and a
    threshold-based search to find highly smooth consecutive pairs.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    N_MAX_PLUS_1 = N_MAX + 1
    SIZE = N_MAX_PLUS_1 + 1  # 1000002
    
    # Optimized LPF sieve using NumPy with reduced memory writes
    lpf = np.zeros(SIZE, dtype=np.int64)
    
    # Sieve: for each prime p, mark multiples. Since we go in increasing order,
    # the last write to each position gives the largest prime factor.
    # Optimization: only iterate over primes, and use efficient slicing.
    p = 2
    while p < SIZE:
        if time.time() > deadline:
            break
        if lpf[p] == 0:  # p is prime
            # Mark all multiples of p starting from p
            # Use step to avoid creating full temporary arrays
            lpf[p:SIZE:p] = p
        p += 1
    
    # Check if sieve completed
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Fallback: use partial results with trial division on a subset
        best_n = N_MIN
        best_val = float('inf')
        
        def lpf_trial(x):
            if x <= 1:
                return 1
            max_pf = 1
            d = 2
            while d * d <= x:
                while x % d == 0:
                    max_pf = max(max_pf, d)
                    x //= d
                d += 1
            return max(max_pf, x)
        
        for n in range(N_MIN, min(N_MIN + 200000, N_MAX + 1)):
            if time.time() > deadline:
                break
            val = max(lpf_trial(n), lpf_trial(n + 1))
            if val < best_val:
                best_val = val
                best_n = n
        return best_n
    
    # Vectorized scan for the best consecutive pair
    n_arr = np.arange(N_MIN, N_MAX + 1, dtype=np.int64)
    max_lpf = np.maximum(lpf[n_arr], lpf[n_arr + 1])
    min_idx = np.argmin(max_lpf)
    best_n = int(n_arr[min_idx])
    
    return best_n
# EVOLVE-BLOCK-END
