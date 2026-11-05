import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    upper = 1000001
    
    # Step 1: Sieve of Eratosthenes to find primes
    is_prime = np.ones(upper + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Mark non-primes
    limit = int(np.sqrt(upper))
    for i in range(2, limit + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    # Extract primes
    primes = np.where(is_prime)[0]
    
    # Check deadline
    if time.time() - start_time > deadline:
        return 10000

    # Step 2: Compute Largest Prime Factor (LPF) for all numbers
    lpf = np.zeros(upper + 1, dtype=np.int32)
    
    # Iterate through primes in increasing order
    for p in primes:
        lpf[p::p] = p
        
    # Check deadline
    if time.time() - start_time > deadline:
        return 10000

    # Step 3: Find n in [10000, 1000000] minimizing max(lpf(n), lpf(n+1))
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[100001:1000002]
    
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    min_idx = np.argmin(max_lpf)
    
    best_n = 10000 + min_idx
    
    return int(best_n)
