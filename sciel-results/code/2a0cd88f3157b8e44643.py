import numpy as np
import math

def construct():
    N = 123457
    
    # Sieve of Eratosthenes to find primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    # Get primes
    primes = np.where(is_prime)[0]
    
    # Compute Largest Prime Factor
    lpf = np.ones(N + 1, dtype=int)
    for p in primes:
        lpf[p::p] = p
        
    # Vectorized search for min max(lpf[a], lpf[b])
    # a ranges from 1 to N-1
    # b = N - a ranges from N-1 to 1
    # lpf_a = lpf[1:N]
    # lpf_b = lpf[N-1:0:-1]
    
    lpf_a = lpf[1:N]
    lpf_b = lpf[N-1:0:-1]
    
    max_lpf = np.maximum(lpf_a, lpf_b)
    
    min_val = np.min(max_lpf)
    idx = np.argmin(max_lpf)
    
    a = idx + 1
    b = N - a
    
    return [a, b]
