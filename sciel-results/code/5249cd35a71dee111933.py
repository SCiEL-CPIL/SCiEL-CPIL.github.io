<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    def largest_factor(v):
        f, d = 1, 2
        while d * d <= v:
            while v % d == 0:
                f, v = d, v // d
            d += 1
        return max(f, v)
    best = (0, 2)
    for p in range(3, 2000, 2):
        if all(p % q for q in range(3, int(p ** 0.5) + 1, 2)):
            k = 0
            while largest_factor(p * p + k + 1) <= p:
                k += 1
            best = max(best, (k, p))
    return {'p': best[1], 'k': best[0]}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

DEADLINE = 145.0  # safety margin before 150s kill

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic Miller-Rabin for n < 3,474,749,660,383
    # For n < 2^64, we need more bases. Since p <= 10^9, p^2+k <= 10^18+200
    # For n < 3.4e12, bases 2,3,5,7,11,13,17 suffice.
    # For n up to 10^18, we need a larger set.
    # Known set for 64-bit: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # Or the smaller set: 2, 325, 9375, 28178, 450775, 9780504, 1795265022
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factorize(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    d = pollard_rho(n)
    factorize(d, factors)
    factorize(n // d, factors)

def largest_prime_factor(n):
    factors = []
    factorize(n, factors)
    return max(factors) if factors else 1

def construct():
    start_time = time.time()
    best = (0, 2)
    
    # First, do a quick search on small primes with the old method to get a baseline
    # Then use the faster factorization to search larger primes
    
    # Generate primes up to a certain limit
    # We can't go too high because p^2 can be up to 10^18, but we need to check p^2+1 .. p^2+k
    # For p ~ 10^9, p^2 is ~ 10^18. Factorization of 10^18 numbers is fast with Pollard-Rho.
    # But we need to find a p that gives a long run.
    
    # Let's search primes in a smart way.
    # We know that small primes might give short runs.
    # Let's try a range of primes.
    
    # Generate primes using sieve up to some limit, then trial division for larger ones
    # Or just iterate and check primality
    
    # Let's try to search in chunks.
    # We'll try primes up to 10^7 first, then if time allows, go higher.
    
    limit = 10**7
    p = 3
    while p < limit:
        if time.time() > DEADLINE:
            break
        if is_prime(p):
            k = 0
            base = p * p
            while k < 200:
                if time.time() > DEADLINE:
                    break
                n = base + k + 1
                # Check if all prime factors of n are <= p
                # We can do this by repeatedly dividing by primes <= p
                # But generating all primes <= p for each p is expensive.
                # Instead, factorize n and check max factor.
                # If max factor > p, break.
                # Optimization: if n <= p, it's trivially true. But n = p^2 + k + 1 > p.
                
                # Quick check: if n is prime, then its largest factor is n > p (since n > p).
                # So if is_prime(n), break immediately.
                if is_prime(n):
                    break
                
                # Otherwise, factorize and check
                # But factorizing every number might be slow if we do it naively.
                # Let's try a hybrid: trial divide by small primes, then Pollard-Rho for the rest.
                
                # Actually, let's just use the factorize function.
                # It should be fast enough for 10^18 numbers.
                
                f = largest_prime_factor(n)
                if f > p:
                    break
                k += 1
            
            if k > best[0]:
                best = (k, p)
        
        p += 2
    
    # If time remains, try larger primes
    # But we need to be careful with the search space.
    # Maybe try some random large primes?
    # Or extend the limit?
    
    if time.time() < DEADLINE - 5:
        # Try some larger primes
        # We can't do a full sieve, but we can check primality
        # Let's try primes in the range [10^7, 10^8]
        # But checking primality for each odd number is slow.
        # Maybe sample some.
        
        # Actually, let's just extend the loop if we have time.
        # But we already broke out. Let's restart with a higher limit if time permits.
        # For now, return what we have.
        pass
    
    return {'p': best[1], 'k': best[0]}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
