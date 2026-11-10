import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # We need powerful parts for n and n+1, so up to MAX_N + 1
    limit = MAX_N + 1
    
    # 1. Sieve of Eratosthenes to find primes up to sqrt(limit)
    sqrt_limit = int(math.isqrt(limit))
    is_prime = [True] * (sqrt_limit + 1)
    if sqrt_limit >= 0:
        is_prime[0] = False
    if sqrt_limit >= 1:
        is_prime[1] = False
        
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, sqrt_limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, sqrt_limit + 1) if is_prime[i]]
    
    # 2. Compute powerful parts
    # pp[n] will store the powerful part of n
    pp = [1] * (limit + 1)
    
    for p in primes:
        p2 = p * p
        if p2 > limit:
            break
            
        # Iterate through multiples of p^2
        # m = k * p^2
        # We need to find the exact exponent e of p in m.
        # m is divisible by p^2.
        # Let m = p^e * r where r is not divisible by p.
        # Since m is multiple of p^2, e >= 2.
        
        # Optimization: 
        # Instead of calculating v_p(m) from scratch for each m,
        # we can iterate k and update.
        # But the "while temp % p == 0" loop is very short.
        
        for m in range(p2, limit + 1, p2):
            # Calculate v_p(m)
            # m is divisible by p^2
            # Let temp = m // p^2
            # e = 2 + v_p(temp)
            temp = m // p2
            e = 2
            while temp % p == 0:
                temp //= p
                e += 1
            pp[m] *= (p ** e)
            
        # Time check
        if time.time() > deadline:
            break
    
    # 3. Find n in [MIN_N, MAX_N] maximizing pp[n] * pp[n+1]
    best_n = MIN_N
    best_val = pp[MIN_N] * pp[MIN_N + 1]
    
    for n in range(MIN_N, MAX_N):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
            
        # Time check
        if time.time() > deadline:
            break
            
    return best_n
