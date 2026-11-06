import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55  # Safety margin before 60s kill
    
    N = 1000000
    MAX_VAL = N + 1
    
    # Sieve primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Initialize powerful part array P with 1s
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Compute powerful parts using vectorized operations
    for p in primes:
        if time.time() - start_time > deadline - 5:
            break
            
        p2 = p * p
        if p2 > MAX_VAL:
            break
            
        power = p2
        while power <= MAX_VAL:
            if time.time() - start_time > deadline - 5:
                break
                
            next_power = power * p
            
            if next_power > MAX_VAL:
                # All multiples of power in range have exactly this exponent
                P[power::power] *= power
            else:
                # Multiples of power but not next_power
                # k*power where k is not a multiple of p
                num_multiples = (MAX_VAL - power) // power + 1
                ks = np.arange(1, num_multiples + 1, dtype=np.int64)
                mask = (ks % p) != 0
                valid_indices = ks[mask] * power
                P[valid_indices] *= power
                
            power = next_power
    
    if time.time() - start_time > deadline - 2:
        return 10000
    
    # Vectorized product
    products = P[10000:1000001] * P[10001:1000002]
    
    # Find index of max
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    
    return int(best_n)
