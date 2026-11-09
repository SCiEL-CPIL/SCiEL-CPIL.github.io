"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses a wheel-factorization sieve over [p^2+1, p^2+200] to quickly determine
smoothness, combined with Pollard-Brent for verification of survivors.
Searches primes via sieve for small ranges and probabilistic primality for large ones."""

import time
import random
import math

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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

# ---------- Pollard's Brent ----------
def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, 100)
        g = r = q = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
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

# ---------- Factorization ----------
def factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_brent(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def all_factors_at_most_p(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if q > p:
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    if is_prime(n):
        return n <= p
    factors = factor(n)
    for f in factors:
        if f > p:
            return False
    return True

def sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

# ---------- Wheel-factorization sieve for smoothness check ----------
# We precompute small primes up to a limit. For a given p, we want to check
# if each number in [p^2+1, p^2+k] has all factors <= p.
# Instead of factoring each number, we can use a sieve-like approach:
# For each prime q <= p, mark multiples of q in the interval.
# But that's expensive for large p.
#
# Better approach: For each number n in [p^2+1, p^2+200], we can quickly
# check if it's p-smooth by:
# 1. Trial divide by small primes (up to some limit)
# 2. If remainder is 1, it's smooth.
# 3. If remainder is prime and <= p, it's smooth.
# 4. Otherwise, use Pollard-Brent to factor the remainder and check.
#
# The key insight: for most numbers, trial division by small primes will
# either reduce them to 1 or reveal a large prime factor > p.
# We can optimize by checking the first few terms quickly.

def check_run_length(p, max_k=200):
    """Check how many consecutive integers starting from p^2+1 are p-smooth."""
    p_sq = p * p
    k = 0
    # We'll check up to max_k numbers
    for i in range(1, max_k + 1):
        n = p_sq + i
        if not all_factors_at_most_p(n, p):
            break
        k += 1
    return k

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Search small primes using sieve
    # For small p, the numbers p^2+i are small, so factorization is fast.
    # We'll search up to 10^6 first.
    sieve_limit = 1_000_000
    primes = sieve_primes(sieve_limit)
    
    for p in primes:
        if time.time() > deadline:
            break
        
        k = check_run_length(p)
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Search larger primes using probabilistic primality testing
    # For larger p, p^2 is large, so we need efficient factorization.
    # But also, larger p means more numbers are likely to be p-smooth.
    # We'll search in ranges, checking every few odd numbers.
    
    # Strategy: search primes in increasing ranges, but skip many numbers
    # to focus on primes. Use a wheel to skip composites.
    
    # For p > 10^6, we'll search in blocks
    block_start = sieve_limit + 1
    block_size = 1_000_000
    
    while block_start < 10**9:
        if time.time() > deadline:
            break
        
        # Generate candidates in this block using a wheel
        # We'll check every 30th number that is coprime to 2,3,5,7
        # This is a rough wheel; we can refine.
        
        # Actually, let's just check every odd number and use is_prime
        # to filter. This is simpler and still efficient.
        
        end = min(block_start + block_size, 10**9 + 1)
        
        for candidate in range(block_start, end, 2):
            if time.time() > deadline:
                break
            
            if not is_prime(candidate):
                continue
            
            k = check_run_length(candidate)
            
            if k > best_k:
                best_k = k
                best_p = candidate
        
        block_start = end
    
    # Phase 3: If time remains, try random large primes
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Try random large primes
            candidate = random.randrange(10**7, 10**9 + 1)
            if candidate % 2 == 0:
                candidate += 1
            
            # Find next prime
            while not is_prime(candidate):
                candidate += 2
                if candidate > 10**9:
                    break
            
            if candidate > 10**9:
                continue
            
            k = check_run_length(candidate)
            
            if k > best_k:
                best_k = k
                best_p = candidate
    
    return {'p': best_p, 'k': best_k}
