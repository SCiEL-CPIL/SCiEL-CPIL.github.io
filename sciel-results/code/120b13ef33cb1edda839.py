import time
import numpy as np

def construct():
    deadline = time.time() + 85
    upper = 1000001  # n+1 can be 1000001
    
    # Use NumPy for faster sieve
    lpf = np.zeros(upper + 1, dtype=np.int32)
    
    for i in range(2, upper + 1):
        if lpf[i] == 0:
            # Vectorized update: set lpf at indices i, 2i, 3i... to i
            # This works because i is the largest prime factor for these numbers?
            # NO. This sets lpf[j] = i for all j divisible by i.
            # Since we iterate i from 2 to upper, the LAST prime factor to touch j
            # is the largest one.
            # So lpf[j] will hold the largest prime factor.
            lpf[i::i] = i
            
            if time.time() > deadline:
                break
    
    # Calculate max LPF for consecutive pairs n, n+1
    # n ranges from 10000 to 1000000
    # lpf[n] and lpf[n+1]
    # Slice 1: lpf[10000 : 1000001] (indices 10000 to 1000000)
    # Slice 2: lpf[10001 : 1000002] (indices 10001 to 1000001)
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find index of minimum value
    min_idx = np.argmin(max_lpf)
    
    best_n = 10000 + min_idx
    
    return int(best_n)
