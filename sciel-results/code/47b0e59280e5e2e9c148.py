import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N = 1000000
    LOW = 10000
    
    # Build smallest prime factor array using NumPy
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    
    limit = int(math.isqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor for each number
    lpf = np.ones(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        x = i
        max_pf = 1
        while x > 1:
            p = int(spf[x])
            if p > max_pf:
                max_pf = p
            while x % p == 0:
                x //= p
        lpf[i] = max_pf
    
    if time.time() > deadline:
        return LOW
    
    # Vectorized scan for best consecutive pair
    window = np.maximum(lpf[LOW:N], lpf[LOW+1:N+1])
    best_idx = int(np.argmin(window))
    return LOW + best_idx
