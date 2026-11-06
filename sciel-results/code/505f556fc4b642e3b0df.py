"""Rewritten program using segmented sieve + Pollard-Brent verification."""
import math
import random
import time

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime_mr(n):
    """Deterministic Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    # Check small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in _MR_BASES:
        if a >= n:
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

def pollard_brent(n):
    """Pollard's Brent cycle detection factorization."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, 10000)
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
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g
        # retry with different parameters

def check_p_smooth(v, p):
    """
    Check if all prime factors of v are <= p.
    Uses Pollard-Brent for factorization to avoid slow trial division for large p.
    """
    if v <= 1:
        return True
    if v <= p:
        return True
    
    # Quick check for small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                     101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    remaining = v
    for sp in small_primes:
        if sp > p:
            break
        if sp * sp > remaining:
            break
        while remaining % sp == 0:
            remaining //= sp
            if remaining == 1:
                return True
    
    if remaining == 1:
        return True
    if remaining <= p:
        return True
    
    # remaining > p. Need to check if it has any prime factor > p.
    # Use recursive factorization with Pollard-Brent.
    
    def _has_large_factor(m, p_limit):
        if m <= 1:
            return False
        if m <= p_limit:
            return False
        if is_prime_mr(m):
            return m > p_limit
        d = pollard_brent(m)
        return _has_large_factor(d, p_limit) or _has_large_factor(m // d, p_limit)
    
    return not _has_large_factor(remaining, p)

def construct():
    best_k = 0
    best_p = 2
    
    # Generate primes up to 10^6 for candidate selection
    SIEVE_LIMIT = 10**6
    sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, SIEVE_LIMIT + 1) if sieve[i]]
    
    def is_prime_simple(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
        
    start_time = time.time()
    time_limit = 115
    
    # Phase 1: Test primes up to 10^6
    for p in small_primes:
        if time.time() - start_time > time_limit:
            break
        k = 0
        for i in range(1, 201):
            v = p * p + i
            if not check_p_smooth(v, p):
                break
            k = i
        if k > best_k:
            best_k = k
            best_p = p
        if k == 200:
            break
            
    # Phase 2: Test larger primes if time permits
    if time.time() - start_time < time_limit:
        # Sample larger primes
        for p in range(1000001, 10000000, 1000):
            if time.time() - start_time > time_limit:
                break
            if not is_prime_mr(p):
                continue
            k = 0
            for i in range(1, 201):
                v = p * p + i
                if not check_p_smooth(v, p):
                    break
                k = i
            if k > best_k:
                best_k = k
                best_p = p
            if k == 200:
                break

    return {'p': best_p, 'k': best_k}
