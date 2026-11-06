"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def largest_prime_factor(x):
    """Compute the largest prime factor of x via exact trial division."""
    if x <= 1:
        return 1
    max_pf = 1
    while x % 2 == 0:
        max_pf = 2
        x //= 2
    f = 3
    while f * f <= x:
        while x % f == 0:
            max_pf = f
            x //= f
        f += 2
    if x > 1:
        max_pf = x
    return max_pf

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    LO, HI = 10000, 1000000
    
    # Precompute primes up to sqrt(1000001) ~ 1000
    # For trial division efficiency
    limit = int(math.isqrt(HI)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def lpf_with_primes(x, primes):
        """Largest prime factor using precomputed primes."""
        if x <= 1:
            return 1
        max_pf = 1
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                max_pf = p
                while x % p == 0:
                    x //= p
        if x > 1:
            max_pf = x
        return max_pf
    
    best_n = LO
    best_val = max(lpf_with_primes(LO, primes), lpf_with_primes(LO + 1, primes))
    
    # Search for smooth consecutive integers
    # Strategy: scan through range, skip numbers with large prime factors
    # A number is "smooth" if its largest prime factor is small
    
    # First pass: quick scan to find candidates
    # We want n and n+1 both to have small largest prime factors
    # The product n(n+1) has lpf = max(lpf(n), lpf(n+1))
    
    # Let's try a multi-stage approach:
    # Stage 1: Scan all n, compute lpf(n) and lpf(n+1)
    # This is O(N * sqrt(N)/log(N)) which might be too slow for 1M
    # Better: use a sieve-like approach or just scan with early termination
    
    # Actually, for n up to 1M, lpf computation via trial division by primes up to 1000
    # is fast enough. Let's just scan.
    
    n = LO
    while n < HI:
        if time.time() > deadline:
            break
        
        # Check n
        ln = lpf_with_primes(n, primes)
        if ln < best_val:
            ln1 = lpf_with_primes(n + 1, primes)
            val = max(ln, ln1)
            if val < best_val:
                best_val = val
                best_n = n
                if best_val <= 95:  # Match or beat observed best
                    break
        
        n += 1
    
    return best_n
# EVOLVE-BLOCK-END
