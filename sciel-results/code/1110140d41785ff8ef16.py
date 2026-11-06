def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 1000000
    limit = N + 1
    
    # 1. Sieve of Eratosthenes using NumPy
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # 2. Largest Prime Factor Sieve using NumPy
    lpf = np.zeros(limit + 1, dtype=np.int32)
    for p in primes:
        lpf[p::p] = p
    
    # 3. Search for best n
    # We want to minimize max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    
    n_range = np.arange(10000, N + 1)
    scores = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    if len(scores) > 0:
        min_idx = np.argmin(scores)
        best_n = int(n_range[min_idx])
    else:
        best_n = 10000
        
    return best_n
