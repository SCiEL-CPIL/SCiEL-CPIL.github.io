import numpy as np
import math

def construct():
    M = 1000000
    limit = M + 1
    
    # Sieve of Eratosthenes to get primes up to limit
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.where(is_prime)[0]
    
    # S[n] will hold the product of primes p where v_p(n) == 1
    # Initialize to 1
    S = np.ones(limit + 1, dtype=np.int64)
    
    for p in primes:
        # Multiply S by p for all multiples of p
        # Indices: p, 2p, 3p, ...
        idx_p = np.arange(p, limit + 1, p)
        S[idx_p] *= p
        
        # Divide S by p for all multiples of p^2
        # This cancels the effect for numbers where v_p(n) >= 2
        # Indices: p^2, 2p^2, ...
        p_sq = p * p
        if p_sq > limit:
            continue
        idx_p_sq = np.arange(p_sq, limit + 1, p_sq)
        S[idx_p_sq] //= p
        
    # Powerful part P(n) = n / S(n)
    # Note: S(n) divides n exactly.
    P = np.arange(limit + 1, dtype=np.int64)
    # Avoid division by zero at 0, but we only care about n >= 10000
    P[1:] = P[1:] // S[1:]
    
    # Find n in [10000, 1000000] maximizing P(n) * P(n+1)
    # Create array of products
    products = P[10000:M+1] * P[10001:M+2]
    
    # Find index of maximum
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    
    return int(best_n)
