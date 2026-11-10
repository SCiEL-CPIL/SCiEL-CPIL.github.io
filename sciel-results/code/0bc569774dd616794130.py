# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    LIMIT = MAX_N + 1  # We need pp[MAX_N+1]
    
    # Build smallest prime factor sieve up to LIMIT
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number in [1, LIMIT]
    # pp[n] = product of p^e for each prime factor p with exponent e >= 2
    pp = np.ones(LIMIT + 1, dtype=np.int64)
    
    # Iterate through primes
    # A number is prime if spf[x] == x
    for p in range(2, LIMIT + 1):
        if spf[p] != p:
            continue
        
        # Check if p^2 > LIMIT
        if p * p > LIMIT:
            break
            
        power = p * p
        next_power = power * p
        
        while power <= LIMIT:
            # Optimization:
            # m = k * power
            # m % next_power != 0  <=>  k % p != 0
            max_k = LIMIT // power
            ks = np.arange(1, max_k + 1)
            mask = (ks % p) != 0
            
            valid_ks = ks[mask]
            target_indices = valid_ks * power
            
            pp[target_indices] *= power
            
            power = next_power
            next_power *= p
            
            # Time check
            if time.time() > deadline - 10:
                break
        
        if time.time() > deadline - 10:
            break
    
    # Find n in [MIN_N, MAX_N] that maximizes pp[n] * pp[n+1]
    # Vectorized
    products = pp[MIN_N:MAX_N+1] * pp[MIN_N+1:MAX_N+2]
    max_idx = np.argmax(products)
    best_n = MIN_N + max_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
