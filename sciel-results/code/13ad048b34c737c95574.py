"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses a direct approach: search for primes p and check consecutive integers p^2+1, ..., p^2+k
for p-smoothness (all prime factors <= p). Uses efficient trial division and Pollard-Brent
with deterministic Miller-Rabin for factorization. Maximizes k within time budget."""

import time
import math
import random

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
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = 128
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

# ---------- Precomputed primes for trial division ----------
_SMALL_PRIMES = None
def get_small_primes(limit=100000):
    """Get precomputed primes up to limit."""
    global _SMALL_PRIMES
    if _SMALL_PRIMES is None:
        _SMALL_PRIMES = sieve_primes(limit)
    return _SMALL_PRIMES

def is_p_smooth(n, p):
    """Check if all prime factors of n are <= p. Returns True if n is 1 or p-smooth."""
    if n <= 1:
        return True
    
    # Try primes up to min(p, sqrt(n)) with trial division
    small_primes = get_small_primes(100000)
    limit = min(p, math.isqrt(n))
    
    for q in small_primes:
        if q > limit:
            break
        if q > p:
            break
        while n % q == 0:
            n //= q
        if n == 1:
            return True
    
    if n == 1:
        return True
    
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
    
    # Use Pollard-Brent to find a factor, then recursively check
    factors = []
    _factor(n, factors, p)
    for f in factors:
        if f > p:
            return False
    return True

def _factor(n, factors, p):
    """Recursively factor n, adding factors to list. Early termination if factor > p."""
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    d = pollard_brent(n)
    _factor(d, factors, p)
    _factor(n // d, factors, p)

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

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to a reasonable limit using sieve
    sieve_limit = 5_000_000
    primes = sieve_primes(sieve_limit)
    
    # Precompute small primes for trial division
    get_small_primes(100000)
    
    for p in primes:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() - start_time > deadline:
                break
            n = p_sq + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If time remains, try some larger primes
    if time.time() - start_time < deadline - 20:
        for _ in range(200):
            if time.time() - start_time > deadline - 5:
                break
            
            lo = max(sieve_limit + 1, 10**6)
            hi = min(10**9, 10**8)
            candidate = random.randrange(lo, hi + 1)
            if candidate % 2 == 0:
                candidate += 1
            
            while not is_prime(candidate):
                candidate += 2
                if candidate > hi:
                    break
            
            if candidate > hi:
                continue
            
            k = 0
            p_sq = candidate * candidate
            while k < 200:
                if time.time() - start_time > deadline - 1:
                    break
                n = p_sq + k + 1
                if not is_p_smooth(n, candidate):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = candidate
    
    return {'p': best_p, 'k': best_k}
