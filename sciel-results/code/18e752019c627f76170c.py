def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 1000000
    limit = N + 1  # We need lpf up to N+1
    
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
    
    # 3. Search for best n using vectorized operations
    # n in [10000, 1000000]
    # We want min(max(lpf[n], lpf[n+1]))
    
    # lpf[n] corresponds to indices 10000 to 1000000
    # lpf[n+1] corresponds to indices 10001 to 1000001
    scores = np.maximum(lpf[10000:1000001], lpf[10001:1000002])
    
    min_score_val = np.min(scores)
    min_score_idx = np.argmin(scores)
    
    best_n = 10000 + min_score_idx
    
    return best_n
