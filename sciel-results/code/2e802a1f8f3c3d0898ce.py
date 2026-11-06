import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1
    
    # 1. Sieve of Eratosthenes to identify primes
    is_prime = np.ones(MAX_VAL + 1, dtype=bool)
    is_prime[0:2] = False
    
    # Standard vectorized sieve
    for i in range(2, int(np.sqrt(MAX_VAL)) + 1):
        if is_prime[i]:
            is_prime[i*i:MAX_VAL+1:i] = False
            
    primes = np.where(is_prime)[0]
    
    # 2. Compute Largest Prime Factor (LPF) for all numbers
    # Initialize LPF array
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    # Iterate through primes in increasing order
    # The last prime to divide a number is its largest prime factor
    for p in primes:
        lpf[p::p] = p
        
    if time.time() > deadline:
        return LO

    # 3. Find n in [LO, HI] minimizing max(LPF(n), LPF(n+1))
    lpf_n = lpf[LO:HI+1]
    lpf_n1 = lpf[LO+1:HI+2]
    
    costs = np.maximum(lpf_n, lpf_n1)
    min_idx = np.argmin(costs)
    best_n = LO + min_idx
    
    return int(best_n)
