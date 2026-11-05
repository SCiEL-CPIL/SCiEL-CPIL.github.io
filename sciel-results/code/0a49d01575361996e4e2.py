"""Find n in [10000, 1000000] maximizing powerful_part(n)*powerful_part(n+1)."""
import time
import math

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    
    N = 1000000
    MAX_N = N + 1  # we need powerful_part up to N+1
    
    # powerful_part[m] = product of p^e for all primes p where e>=2 in factorization of m
    # Initialize to 1
    powerful_part = [1] * (MAX_N + 1)
    
    # Sieve to find primes
    is_prime = [True] * (MAX_N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, MAX_N + 1) if is_prime[i]]
    
    # For each prime p, compute p^k for k>=2, and multiply p^k into powerful_part[m]
    # for every multiple m of p^k
    for p in primes:
        p2 = p * p
        if p2 > MAX_N:
            break
        # p^2, p^3, p^4, ... up to MAX_N
        pk = p2
        while pk <= MAX_N:
            # For every multiple m of pk, multiply powerful_part[m] by pk
            for m in range(pk, MAX_N + 1, pk):
                powerful_part[m] *= pk
            pk *= p
            # Time check
            if time.time() > deadline:
                # Fall through to best found so far (partial computation)
                # But we need at least [10000, 1000001] computed
                # If we're still in the sieve, we might not have enough
                pass
    
    # Now find the best n in [10000, 1000000] maximizing powerful_part[n] * powerful_part[n+1]
    best_n = 10000
    best_val = powerful_part[10000] * powerful_part[10001]
    
    for n in range(10000, 1000001):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
