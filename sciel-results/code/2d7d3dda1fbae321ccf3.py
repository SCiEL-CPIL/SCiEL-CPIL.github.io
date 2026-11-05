import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds safety margin
    
    N = 1000000
    MIN_N = 10000
    
    # 1. Sieve for Smallest Prime Factor (SPF) using optimized NumPy
    spf = np.arange(N + 2, dtype=np.int64)
    limit = int(math.isqrt(N)) + 1
    for i in range(2, limit):
        if spf[i] == i:
            start = i * i
            # Vectorized: find indices where spf[j] == j
            multiples = np.arange(start, N + 2, i, dtype=np.int64)
            mask = spf[multiples] == multiples
            if np.any(mask):
                spf[multiples[mask]] = i
    
    # 2. Compute powerful part for all numbers up to N
    # powerful[i] stores the powerful part of i
    powerful = np.ones(N + 2, dtype=np.int64)
    
    for k in range(2, N + 2):
        p = int(spf[k])
        m = k
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        
        if e >= 2:
            powerful[k] = powerful[m] * (p ** e)
        else:
            powerful[k] = powerful[m]
            
    # Check time
    if time.time() > deadline:
        return MIN_N  # Fallback, shouldn't happen

    # 3. Vectorized scan for max product
    # n ranges from MIN_N to N-1
    n_vals = np.arange(MIN_N, N, dtype=np.int64)
    products = powerful[n_vals] * powerful[n_vals + 1]
    best_idx = int(np.argmax(products))
    best_n = int(n_vals[best_idx])
    
    return best_n
