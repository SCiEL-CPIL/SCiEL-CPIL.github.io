"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _sieve_primes(limit):
    """Return list of primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def _is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.3e24."""
    if n < 2:
        return False
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Write n-1 = d * 2^r
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # Bases sufficient for n < 3.3e24
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _pollard_brent(n, random_seed=None):
    """Pollard's Brent cycle-finding factorization."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = random
    while True:
        y = rng.randrange(2, n - 1)
        c = rng.randrange(1, n - 1)
        m = 128
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

def _factorize(n, small_primes, small_prime_limit):
    """Factorize n, returning list of prime factors (with multiplicity)."""
    factors = []
    for p in small_primes:
        if p * p > n:
            break
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1:
        if _is_prime_mr(n):
            factors.append(n)
        else:
            # Use Pollard-Brent for composite
            stack = [n]
            while stack:
                m = stack.pop()
                if m == 1:
                    continue
                if _is_prime_mr(m):
                    factors.append(m)
                else:
                    d = _pollard_brent(m)
                    stack.append(d)
                    stack.append(m // d)
    return factors

def _largest_prime_factor(n, small_primes, small_prime_limit):
    """Return the largest prime factor of n."""
    for p in small_primes:
        if p * p > n:
            break
        while n % p == 0:
            n //= p
    if n > 1:
        if _is_prime_mr(n):
            return n
        else:
            # Need full factorization to find largest
            factors = _factorize(n, small_primes, small_prime_limit)
            return max(factors)
    # n became 1, all factors were in small_primes
    # But we lost track; re-do
    return _largest_prime_factor_slow(n_original, small_primes) if False else _lpf_helper(n, small_primes)

def _lpf_helper(n, small_primes):
    """Helper: find largest prime factor after trial division."""
    # This is called when n > 1 after trial division
    if _is_prime_mr(n):
        return n
    factors = _factorize(n, small_primes, 10000)
    return max(factors)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin: 10s before 150s kill
    
    # Generate small primes for trial division
    small_prime_limit = 200000
    small_primes = _sieve_primes(small_prime_limit)
    small_prime_set = set(small_primes)
    
    # Precompute primes up to a reasonable limit for candidate p
    # We'll search primes p in increasing order
    # For each prime p, check how many of p^2+1, ..., p^2+k are p-smooth
    
    best_k = 1
    best_p = 3
    
    # We need to check primes p. The key insight: p must be prime and <= 10^9.
    # For small p, p^2 is small, so p^2+1 through p^2+k are small numbers.
    # The constraint is that all prime factors of p^2+i must be <= p.
    # This means p^2+i must be p-smooth.
    
    # Strategy: iterate over candidate primes p, and for each, check the run length.
    # To speed up, we can pre-sieve: for each small prime q > p, check if any
    # of p^2+1..p^2+200 is divisible by q. If yes, that position fails.
    # But that's complex. Instead, just factorize each p^2+i and check max factor.
    
    # Let's iterate over primes p. We'll use a segmented approach for generating primes.
    # Start with small primes, then go larger.
    
    # First, try all primes up to some limit using the sieve
    max_p_search = 1000000  # start with primes up to 1M
    primes_up_to = _sieve_primes(max_p_search)
    
    for p in primes_up_to:
        if time.time() > deadline:
            break
        if p > 10**9:
            break
        
        # Check run length for this p
        k = 0
        p_sq = p * p
        for i in range(1, 201):
            if time.time() > deadline:
                break
            n = p_sq + i
            # Check if n is p-smooth: all prime factors <= p
            # Quick check: try dividing by primes up to p
            # But p can be up to 1M, and we have small_primes up to 200K
            # For n = p^2 + i, the smallest prime factor > p would make it fail
            
            # Efficient check: factorize n and verify all factors <= p
            # Use trial division by small primes first
            temp = n
            for q in small_primes:
                if q > p:
                    break
                if q * q > temp:
                    break
                while temp % q == 0:
                    temp //= q
            if temp > 1:
                # temp has a factor > p or is prime > p
                # Check if temp is prime
                if _is_prime_mr(temp):
                    if temp > p:
                        break  # found a prime factor > p
                else:
                    # temp is composite, find its factors
                    # If any factor > p, we fail
                    factors = _factorize(temp, small_primes, small_prime_limit)
                    if any(f > p for f in factors):
                        break
            # All factors of n are <= p, so n is p-smooth
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
