import time
import numpy as np

def construct():
    # Use a slightly larger deadline to be safe, though NumPy is fast
    deadline = time.time() + 80
    
    max_val = 1000001  # n+1 can be up to 1000001
    
    # Sieve for largest prime factor using NumPy for speed
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    for p in range(2, max_val + 1):
        if lpf[p] == 0:  # p is prime
            lpf[p::p] = p
    
    # Vectorized scan for n in [10000, 1000000]
    # We need max(lpf[n], lpf[n+1])
    # n ranges from 10000 to 1000000
    # lpf[n] ranges from lpf[10000] to lpf[1000000]
    # lpf[n+1] ranges from lpf[10001] to lpf[1000001]
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    vals = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(vals)
    
    # The index in vals corresponds to n = 10000 + min_idx
    best_n = 10000 + min_idx
    
    # If time is running out (unlikely), we still have the best_n
    if time.time() > deadline:
        pass # Just return what we have
        
    return best_n
