import time
import numpy as np

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    max_val = 1000001  # n+1 can be up to 1000001
    
    # Sieve for largest prime factor using NumPy for speed
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    for p in range(2, max_val + 1):
        if lpf[p] == 0:  # p is prime
            lpf[p::p] = p
    
    # Vectorized scan for n in [10000, 1000000]
    # We need max(lpf[n], lpf[n+1])
    # n ranges from 10000 to 1000000
    # lpf[n] corresponds to indices 10000 to 1000000
    # lpf[n+1] corresponds to indices 10001 to 1000001
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    vals = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(vals)
    
    # The index in vals corresponds to n = 10000 + min_idx
    # Ensure the result is a standard Python integer
    best_n = int(10000 + min_idx)
    
    return best_n
