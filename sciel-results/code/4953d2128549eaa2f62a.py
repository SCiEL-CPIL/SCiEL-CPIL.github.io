import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 1000000
    max_val = N + 1  # we need pp up to N+1
    
    # Array to store powerful parts, 1-indexed
    pp = np.ones(max_val + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to get primes up to sqrt(N+1)
    limit = int(math.isqrt(max_val))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For each prime, compute its contribution to powerful parts
    # P(n) = prod p^e if e>=2 else 1
    # We can decompose this into:
    # Multiply by p^2 for all n divisible by p^2
    # Multiply by p   for all n divisible by p^3
    # Multiply by p   for all n divisible by p^4
    # ...
    for p in primes:
        p2 = p * p
        if p2 > max_val:
            continue
        
        # Step 1: Multiply by p^2 for multiples of p^2
        pp[p2::p2] *= p2
        
        # Step 2: Multiply by p for multiples of p^3, p^4, ...
        pe = p2 * p  # p^3
        while pe <= max_val:
            pp[pe::pe] *= p
            pe *= p
            
        # Check deadline
        if time.time() > deadline:
            break
    
    # Now find the max product pp[n] * pp[n+1] for n in [10000, 1000000]
    # Use numpy for vectorized computation
    products = pp[10000:N+1] * pp[10001:N+2]
    idx = np.argmax(products)
    best_n = 10000 + idx
    
    return int(best_n)
