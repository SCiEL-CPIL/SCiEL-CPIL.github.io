import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    N = 1000000
    MIN_N = 10000
    
    # Precompute largest prime factor for all numbers up to N using NumPy
    lpf = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes variant for Largest Prime Factor
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            lpf[i::i] = i
    
    # Check time after sieve
    if time.time() > deadline:
        return 10000
    
    # Vectorized search: compute max(lpf[n], lpf[n+1]) for all n in range
    # n ranges from MIN_N to N-1 (so n+1 <= N)
    n_range = np.arange(MIN_N, N)
    lpf_n = lpf[n_range]
    lpf_n1 = lpf[n_range + 1]
    max_lpf = np.maximum(lpf_n, lpf_n1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf)
    min_lpf_max = max_lpf[min_idx]
    best_n = n_range[min_idx]
    
    # Time check
    if time.time() > deadline:
        return int(best_n)
    
    return int(best_n)
