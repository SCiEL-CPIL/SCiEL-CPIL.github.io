import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    LIMIT = 1000000
    MAX_VAL = LIMIT + 1
    
    # lpf[i] stores the largest prime factor of i
    lpf = [0] * (MAX_VAL + 1)
    
    # Sieve to compute largest prime factors
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:
            # p is prime
            for multiple in range(p, MAX_VAL + 1, p):
                lpf[multiple] = p
                
    # Convert to numpy array for vectorized operations
    lpf_arr = np.array(lpf)
    
    # We need max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    # lpf[n] is lpf_arr[n]
    # lpf[n+1] is lpf_arr[n+1]
    # Range of n: 10000 to 1000000
    # lpf_arr[10000:1000001] corresponds to lpf[n]
    # lpf_arr[10001:1000002] corresponds to lpf[n+1]
    
    if time.time() > deadline:
        return 10000 # Fallback
        
    # Vectorized max
    # Note: lpf_arr is 1-indexed effectively for our purposes, but 0-indexed in array
    # n=10000 -> index 10000
    # n=1000000 -> index 1000000
    # n+1=1000001 -> index 1000001
    
    slice_n = lpf_arr[10000:1000001]
    slice_np1 = lpf_arr[10001:1000002]
    
    max_factors = np.maximum(slice_n, slice_np1)
    
    min_val = np.min(max_factors)
    min_idx = np.argmin(max_factors)
    
    best_n = 10000 + min_idx
    
    return int(best_n)
