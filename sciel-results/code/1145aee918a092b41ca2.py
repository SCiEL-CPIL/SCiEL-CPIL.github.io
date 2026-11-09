import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    MAX_N = 1000001
    
    # 1. Generate Smallest Prime Factor (SPF)
    # Initialize with identity
    spf = np.arange(MAX_N + 1, dtype=np.int32)
    spf[0] = 0
    spf[1] = 1
    
    # Sieve
    limit = int(math.isqrt(MAX_N))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            # Mark multiples of i starting from i*i
            # We want to set spf[j] to i if it hasn't been set to a smaller prime
            # Since we iterate i in increasing order, the first prime that divides j is the smallest.
            # However, the standard SPF sieve updates spf[j] = i only if spf[j] is still j (or 0).
            # With np.arange init, spf[j] starts as j.
            # So we check if spf[j] == j.
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # 2. Compute Largest Prime Factor (LPF)
    lpf = np.zeros(MAX_N + 1, dtype=np.int32)
    lpf[1] = 1
    
    # Compute LPF using SPF
    # lpf[i] = max(spf[i], lpf[i // spf[i]])
    # This works because i // spf[i] < i, so lpf[i // spf[i]] is already computed.
    
    # We can try to vectorize this partially, but the dependency i//spf[i] makes it tricky.
    # A simple loop is O(N) and fast enough.
    
    for i in range(2, MAX_N + 1):
        p = spf[i]
        if p == i:
            lpf[i] = i
        else:
            lpf[i] = max(p, lpf[i // p])
            
    if time.time() > deadline:
        return 10000
        
    # 3. Find best n
    start_idx = 10000
    end_idx = 1000001 # exclusive
    
    max_lpf = np.maximum(lpf[start_idx:end_idx], lpf[start_idx+1:end_idx+1])
    
    best_idx = np.argmin(max_lpf)
    best_n = start_idx + best_idx
    
    return int(best_n)
