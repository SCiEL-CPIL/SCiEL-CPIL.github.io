import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    
    MAX_VAL = 1000000
    # We need LPF for n and n+1, so up to 1000001
    MAX_LPF = 1000001
    
    # Sieve to find largest prime factor
    # Using NumPy for speed
    lpf = np.zeros(MAX_LPF + 1, dtype=np.int32)
    
    for i in range(2, MAX_LPF + 1):
        if lpf[i] == 0:
            # i is prime
            lpf[i::i] = i
            
    if time.time() - start_time > DEADLINE:
        return 10000

    # Convert to list for faster indexing in the scan loop? 
    # Or just use numpy array indexing which is also fast.
    # lpf[n] and lpf[n+1] are O(1).
    
    best_n = -1
    best_score = float('inf')
    
    # We can vectorize the scan too?
    # lpf[10000:1000000] and lpf[10001:1000001]
    # max of these two arrays element-wise
    # then argmin
    
    # n ranges from 10000 to 1000000
    # lpf_n = lpf[10000:1000001]  # indices 10000 to 1000000
    # lpf_np1 = lpf[10001:1000002] # indices 10001 to 1000001
    
    # Let's check sizes
    # lpf has size 1000002 (0 to 1000001)
    # n = 10000 -> lpf[10000], lpf[10001]
    # n = 1000000 -> lpf[1000000], lpf[1000001]
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    # Element-wise max
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find index of minimum
    min_idx = np.argmin(max_lpf)
    
    best_n = 10000 + min_idx
    best_score = max_lpf[min_idx]
    
    return best_n
