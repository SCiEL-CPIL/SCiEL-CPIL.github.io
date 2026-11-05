import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds safety margin
    
    N = 1000000
    MIN_N = 10000
    
    # 1. Sieve for Smallest Prime Factor (SPF) using NumPy for speed
    spf = np.arange(N + 2, dtype=np.int64)
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            spf[i*i::i] = np.where(spf[i*i::i] == spf[i*i::i], spf[i*i::i], spf[i*i::i])
            # Only set if not already set
            mask = spf[i*i::i] == i
            if np.any(mask):
                spf[i*i::i][mask] = i

    # 2. Compute powerful part for all numbers up to N
    powerful = np.ones(N + 2, dtype=np.int64)
    
    for k in range(2, N + 2):
        p = spf[k]
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
        return MIN_N

    # 3. Vectorized scan for max product
    # n ranges from MIN_N to N-1
    n_vals = np.arange(MIN_N, N, dtype=np.int64)
    products = powerful[n_vals] * powerful[n_vals + 1]
    best_idx = np.argmax(products)
    best_n = int(n_vals[best_idx])
    best_val = int(products[best_idx])

    return best_n
