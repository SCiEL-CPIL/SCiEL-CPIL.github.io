# EVOLVE-BLOCK-START
import math
import numpy as np

def construct():
    MAX_VAL = 1000001
    limit = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        # Multiply by p^2 for all multiples of p^2
        P[p2::p2] *= p2
        
        pk = p2 * p
        while pk <= MAX_VAL:
            # Multiply by p for all multiples of p^3, p^4, etc.
            P[pk::pk] *= p
            pk *= p
            
    # We want to maximize P[n] * P[n+1] for n in [10000, 1000000]
    # P[n] corresponds to index n
    # P[n+1] corresponds to index n+1
    # n ranges from 10000 to 1000000
    # So we need P[10000]...P[1000000] and P[10001]...P[1000001]
    
    vals = P[10000:1000001] * P[10001:1000002]
    best_offset = np.argmax(vals)
    best_n = 10000 + best_offset
    
    return int(best_n)

# EVOLVE-BLOCK-END
