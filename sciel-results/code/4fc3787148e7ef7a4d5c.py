import numpy as np
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    # 1. Sieve of Eratosthenes to find primes up to N-1
    # We only need LPF for numbers up to N-1
    limit = N - 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
            
    primes = np.where(is_prime)[0]
    
    # 2. Compute Largest Prime Factor (LPF) for all numbers up to N-1
    lpf = np.ones(limit + 1, dtype=np.int64)
    
    for p in primes:
        lpf[p::p] = p
        
    # Check time (unlikely to be needed for N=123457)
    if time.time() - start_time > deadline:
        return (1, N - 1)

    # 3. Find the split a + b = N that minimizes max(P(a), P(b))
    a = np.arange(1, N, dtype=np.int64)
    b = N - a
    
    cost = np.maximum(lpf[a], lpf[b])
    
    min_idx = np.argmin(cost)
    
    best_a = a[min_idx]
    best_b = b[min_idx]
    
    return (int(best_a), int(best_b))
