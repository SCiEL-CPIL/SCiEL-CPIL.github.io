"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9)."""

import time
import random
import math

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
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
        x = random.randrange(2, n)
        y = x
        c = random.randrange(1, n)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_rho(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def max_prime_factor(n):
    """Return the largest prime factor of n, or 1 if n==1."""
    if n <= 1:
        return 1
    if is_prime(n):
        return n
    facs = factor(n)
    return max(facs)

def is_p_smooth(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    if is_prime(n):
        return n <= p
    facs = factor(n)
    return all(f <= p for f in facs)

def construct():
    deadline = time.time() + 140  # safety margin before 150s
    
    best_k = 0
    best_p = 0
    
    # Precompute small primes for trial division speedup
    small_primes = []
    for i in range(2, 10000):
        if is_prime(i):
            small_primes.append(i)
    
    # Search over primes p. Start with small primes, then larger.
    # The key insight: for p^2 + i to be p-smooth, it's a strong condition.
    # Let's search a range of primes.
    
    # First, try primes up to a reasonable limit
    # Generate primes efficiently
    def primes_up_to(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, v in enumerate(sieve) if v]
    
    # Try primes in increasing order, but also try some larger primes
    # Start with primes up to 100000
    primes_list = primes_up_to(500000)
    
    # Also add some larger primes to check
    # For larger p, p^2 + i is ~ p^2, and we need all factors <= p
    # This means p^2 + i must be composite with factors in [2, p]
    
    for p in primes_list:
        if time.time() > deadline:
            break
        if p < 2:
            continue
        
        k = 0
        for i in range(1, 201):
            n = p * p + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we have time, try some larger primes
    if time.time() < deadline - 5:
        # Try primes in ranges: [100000, 1000000], etc.
        for start in [100000, 1000000, 10000000, 100000000]:
            if time.time() > deadline - 5:
                break
            end = min(start * 10, 1000000000)
            # Sample some primes in this range
            # Use a simple sieve for a small window
            window_size = min(100000, end - start)
            if window_size <= 0:
                continue
            sieve_start = start
            sieve_end = start + window_size
            sieve = [True] * (window_size)
            for sp in small_primes:
                if sp * sp > sieve_end:
                    break
                # Find first multiple of sp >= sieve_start
                first = ((sieve_start + sp - 1) // sp) * sp
                for j in range(first - sieve_start, window_size, sp):
                    sieve[j] = False
            # Check primality of candidates
            for i in range(window_size):
                if time.time() > deadline - 5:
                    break
                cand = sieve_start + i
                if cand < 2:
                    continue
                if sieve[i] and is_prime(cand):
                    # Check this prime
                    k = 0
                    for ii in range(1, 201):
                        n = cand * cand + ii
                        if is_p_smooth(n, cand):
                            k += 1
                        else:
                            break
                    if k > best_k:
                        best_k = k
                        best_p = cand
    
    if best_p == 0:
        best_p = 2
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
