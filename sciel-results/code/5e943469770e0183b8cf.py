import time
import math
import random
from functools import lru_cache

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
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

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
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
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
            if g == n:
                continue
        return g

# ---------- Factorization ----------
@lru_cache(maxsize=None)
def _factor(n):
    if n == 1:
        return ()
    if _is_prime_mr(n):
        return (n,)
    d = _pollard_brent(n)
    return _factor(d) + _factor(n // d)

def _all_factors_at_most_p(n, p):
    if n <= 1:
        return True
    # Quick check: if n is prime, it must be <= p
    if _is_prime_mr(n):
        return n <= p
    
    # Trial division by small primes to reduce n
    # We only care if ANY factor is > p
    # If we find a factor > p, return False
    # If we reduce n to 1 or a prime <= p, return True
    # If n becomes composite, factor it and check
    
    # Optimize: if n <= p, all factors must be <= p
    if n <= p:
        return True
        
    # Check small prime factors
    for q in _MR_BASES:
        if q > p:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
            if n == 1:
                return True
            if n <= p:
                return True
            # Continue checking n
    
    if n == 1:
        return True
    if n <= p:
        return True
        
    # If n is prime, check if <= p
    if _is_prime_mr(n):
        return n <= p
        
    # Factor n
    factors = _factor(n)
    for f in factors:
        if f > p:
            return False
    return True

def sieve_primes(limit):
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
    deadline = start_time + 140  # Safety margin

    best_k = 0
    best_p = 2

    # Precompute small primes for trial division in _all_factors_at_most_p
    # This is already done in _MR_BASES, but we might want more.
    # For now, rely on _MR_BASES and Pollard-Brent.

    # Search small primes first to get a baseline
    small_primes = sieve_primes(10000)
    for p in small_primes:
        if time.time() > deadline:
            break
        k = 0
        p_sq = p * p
        for j in range(1, 201):
            n = p_sq + j
            if _all_factors_at_most_p(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Search larger primes in chunks
    # We will search up to 10^7 or more if time permits
    chunk_size = 100000
    start = 10000
    while start < 10**7:
        if time.time() > deadline:
            break
        end = min(start + chunk_size, 10**7)
        # Generate primes in [start, end]
        # Simple trial division or sieve
        # Sieve is faster
        is_p = bytearray(b'\x01') * (end - start + 1)
        # Sieve out composites
        # We need primes up to sqrt(end)
        limit = int(math.isqrt(end)) + 1
        small_primes_for_sieve = sieve_primes(limit)
        
        for sp in small_primes_for_sieve:
            # Find first multiple of sp in [start, end]
            first = ((start + sp - 1) // sp) * sp
            if first == sp:
                first += sp
            for multiple in range(first, end + 1, sp):
                is_p[multiple - start] = 0
                
        for i in range(end - start + 1):
            if time.time() > deadline:
                break
            if is_p[i]:
                p = start + i
                if p < 2:
                    continue
                # Check run length
                k = 0
                p_sq = p * p
                for j in range(1, 201):
                    n = p_sq + j
                    if _all_factors_at_most_p(n, p):
                        k += 1
                    else:
                        break
                if k > best_k:
                    best_k = k
                    best_p = p
        start = end

    # If time remains, search even larger primes randomly
    if time.time() < deadline - 10:
        for _ in range(1000):
            if time.time() > deadline:
                break
            p = random.randint(10**7, 10**9)
            if p % 2 == 0:
                p += 1
            while not _is_prime_mr(p):
                p += 2
            k = 0
            p_sq = p * p
            for j in range(1, 201):
                n = p_sq + j
                if _all_factors_at_most_p(n, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
