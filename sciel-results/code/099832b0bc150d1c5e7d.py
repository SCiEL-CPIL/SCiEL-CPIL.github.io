def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N_MAX = 1000000
    limit = N_MAX + 1
    
    lpf = np.zeros(limit + 1, dtype=np.int32)
    
    # Sieve for largest prime factor with vectorized inner loop
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            # i is prime, update all multiples using numpy
            multiples = np.arange(i, limit + 1, i, dtype=np.int32)
            lpf[multiples] = i
        
        if i % 50000 == 0 and time.time() > deadline:
            break
    
    n_start = 10000
    n_end = N_MAX
    
    n_indices = np.arange(n_start, n_end + 1, dtype=np.int32)
    vals = np.maximum(lpf[n_indices], lpf[n_indices + 1])
    min_idx = np.argmin(vals)
    best_n = int(n_indices[min_idx])
    
    return best_n
