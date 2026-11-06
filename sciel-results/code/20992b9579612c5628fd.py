def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 1000000
    limit = N + 1
    
    # 1. Sieve of Eratosthenes using NumPy
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    # Sieve
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    if time.time() > deadline:
        return 10000

    primes = np.where(is_prime)[0]
    
    # 2. Largest Prime Factor Sieve
    # Use int32 to save memory and potentially speed up operations
    lpf = np.zeros(limit + 1, dtype=np.int32)
    
    for p in primes:
        if p > limit:
            break
        lpf[p::p] = p
        
    if time.time() > deadline:
        return 10000

    # 3. Search for best n
    # Use int32 for n_range to match lpf dtype and save memory
    n_range = np.arange(10000, N + 1, dtype=np.int32)
    
    # Vectorized max
    # lpf[n_range] and lpf[n_range + 1]
    # Note: n_range + 1 will be int32 if N < 2^31, which it is.
    scores = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    if len(scores) > 0:
        min_idx = np.argmin(scores)
        best_n = int(n_range[min_idx])
    else:
        best_n = 10000
        
    return best_n
