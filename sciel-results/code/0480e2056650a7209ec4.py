import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 1000000
    # We need P(n) for n up to N+1
    size = N + 2
    P = [1] * size
    
    # Sieve of Eratosthenes to find primes up to sqrt(N)
    # sqrt(1000000) = 1000
    limit = int(math.isqrt(N))
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Calculate powerful parts
    for p in primes:
        p2 = p * p
        # Iterate through multiples of p
        # Optimization: Only start from p2? No, we need to check exponent for all multiples.
        # But if m < p2, exponent is 0 or 1.
        # So we can start from p2?
        # If m is a multiple of p, but m < p2, then m is p itself (e=1).
        # So yes, we can start from p2.
        for m in range(p2, size, p):
            x = m
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                P[m] *= (p ** e)
                
    if time.time() > deadline:
        return 10000

    best_n = 10000
    max_val = P[10000] * P[10001]
    
    # Search for max product
    for n in range(10000, N):
        val = P[n] * P[n+1]
        if val > max_val:
            max_val = val
            best_n = n
            
        if time.time() > deadline:
            break
            
    return best_n
