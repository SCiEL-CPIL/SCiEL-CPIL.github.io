import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    max_val = 1000001
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # Sieve for largest prime factor
    # 2 is prime
    lpf[2::2] = 2
        
    # Odd primes
    for p in range(3, max_val + 1, 2):
        if lpf[p] == 0:  # p is prime
            lpf[p::p] = p

    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # Using vectorized NumPy operations for speed
    n_range = np.arange(10000, 1000001)
    max_lpf = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    best_idx = np.argmin(max_lpf)
    best_n = int(n_range[best_idx])
    
    return best_n
