def construct():
    import time
    import numpy as np
    import math
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 1000001
    # Initialize largest prime factor array
    lpf = np.zeros(N + 2, dtype=np.int32)
    
    # Sieve for largest prime factor
    # i is the current candidate for prime
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
            if time.time() > deadline:
                break

    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # Vectorized search
    n_range = np.arange(10000, 1000001)
    vals = np.maximum(lpf[n_range], lpf[n_range + 1])
    min_idx = np.argmin(vals)
    best_n = int(n_range[min_idx])
    
    return best_n
