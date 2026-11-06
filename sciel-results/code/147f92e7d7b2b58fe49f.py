import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    min_n = 10000
    max_n = 1000000
    limit = max_n + 1
    N = limit + 1

    # Sieve of Eratosthenes to find primes up to sqrt(limit)
    sqrt_limit = int(np.sqrt(limit)) + 1
    is_prime = np.ones(sqrt_limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(np.sqrt(sqrt_limit)) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = False
    primes = np.where(is_prime)[0]

    # Largest prime factor array
    lpf = np.zeros(N, dtype=np.int32)
    
    for p in primes:
        if p * p > limit:
            break
        # Mark multiples of p: lpf[m] = p if p is the largest prime factor so far
        # We iterate from p*p to limit, step p
        for m in range(p*p, limit + 1, p):
            lpf[m] = p
    
    # For numbers that are prime themselves (lpf still 0), set lpf to themselves
    # But we only need to handle numbers that weren't marked by any prime <= sqrt(limit)
    # Actually, the above loop marks composites. Primes remain 0.
    # We need lpf[x] = x for primes x.
    # But we can't easily identify which 0s are primes vs 1.
    # Let's just do a final pass: if lpf[i] == 0 and i >= 2, then i is prime, lpf[i] = i.
    # This is O(N) but fast in numpy.
    # However, the loop above already set lpf for composites.
    # Let's check: for a prime p, no smaller prime divides it, so lpf[p] remains 0.
    # For a composite c, it has a prime factor <= sqrt(c) <= sqrt(limit), so it gets marked.
    # So: lpf[i] == 0 implies i is prime (or i < 2).
    
    # Set lpf for primes
    mask = (lpf == 0) & (np.arange(N) >= 2)
    lpf[mask] = np.arange(N)[mask]
    
    # Compute M(n) = max(lpf[n], lpf[n+1]) for n in [min_n, max_n]
    left = lpf[min_n:max_n + 1]
    right = lpf[min_n + 1:max_n + 2]
    
    max_factors = np.maximum(left, right)
    min_idx = np.argmin(max_factors)
    
    best_n = min_n + min_idx
    
    return int(best_n)
