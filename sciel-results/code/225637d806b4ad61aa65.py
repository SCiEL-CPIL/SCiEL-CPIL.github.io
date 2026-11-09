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
    # Trial divide by small primes <= p
    for q in _MR_BASES:
        if q > p:
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    if _is_prime_mr(n):
        return n <= p
    factors = _factor(n)
    return all(f <= p for f in factors)

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
    deadline = start_time + 140  # safety margin before 150s kill

    best_k = 0
    best_p = 2

    # Precompute small primes up to a limit for fast trial division.
    SMALL_LIMIT = 2_000_000
    small_primes = sieve_primes(SMALL_LIMIT)
    small_prime_set = set(small_primes)

    # Precompute QR bitsets for small primes.
    # For each small prime q, compute a bitset of length 201 where bit j is set
    # if there exists an integer r such that r^2 ≡ -j (mod q), i.e., -j is a QR mod q.
    # For a candidate p, if p mod q = r and bit j is set, then q | (p^2 + j).
    # We only care about q > p (since q <= p is fine for p-smoothness).
    
    QR_LIMIT = 50_000  # Precompute QR bitsets for primes up to this limit
    qr_bitsets = {}
    for q in small_primes:
        if q > QR_LIMIT:
            break
        # Compute which j in [1, 200] have -j as a QR mod q
        # By iterating r from 0 to q-1, mark j = (q - r^2 % q) % q
        bitset = [0] * 201
        for r in range(q):
            r2 = (r * r) % q
            j = (q - r2) % q
            if 1 <= j <= 200:
                bitset[j] = 1
        qr_bitsets[q] = bitset

    def quick_smooth_run(p, max_k=200):
        """Quick check using QR bitsets for small primes q in (p, QR_LIMIT]."""
        k = 0
        p_sq = p * p
        p_mod_cache = {}
        
        for i in range(1, max_k + 1):
            n = p_sq + i
            # First, trial divide by small primes <= p
            temp_n = n
            for sp in small_primes:
                if sp > p:
                    break
                if sp * sp > temp_n:
                    break
                while temp_n % sp == 0:
                    temp_n //= sp
            if temp_n == 1:
                k += 1
                continue
            if temp_n <= p:
                k += 1
                continue
            # temp_n > p: check if temp_n is prime
            if _is_prime_mr(temp_n):
                break
            # Composite and > p: need to check if all factors <= p
            # Quick check using QR bitsets
            # For each small prime q in (p, QR_LIMIT], check if q divides n
            # i.e., if p^2 + i ≡ 0 (mod q), i.e., if (p mod q)^2 ≡ -i (mod q)
            bad = False
            for q in small_primes:
                if q <= p:
                    continue
                if q > QR_LIMIT:
                    break
                if q * q > temp_n:
                    break
                # Check if q divides n
                if n % q == 0:
                    # q > p and q | n, so n has a prime factor > p
                    bad = True
                    break
            if bad:
                break
            # If we get here, temp_n might be composite with all factors <= p
            # or we missed some factor. Do a full check.
            if not _all_factors_at_most_p(n, p):
                break
            k += 1
        
        return k

    def verify_run(p, k):
        """Full verification that all of p^2+1..p^2+k are p-smooth."""
        p_sq = p * p
        for i in range(1, k + 1):
            n = p_sq + i
            if not _all_factors_at_most_p(n, p):
                return False
        return True

    # Phase 1: Search small primes up to 10^6
    search_limit = 1_000_000
    primes_to_test = [p for p in small_primes if p <= search_limit]

    for p in primes_to_test:
        if time.time() > deadline:
            break
        k = quick_smooth_run(p, 200)
        if k > best_k:
            if verify_run(p, k):
                best_k = k
                best_p = p

    # Phase 2: Search larger primes using primality testing
    if time.time() < deadline - 15:
        # Search in chunks
        chunk_size = 100_000
        for start in range(1_000_000, 50_000_000, chunk_size):
            if time.time() > deadline:
                break
            end = min(start + chunk_size, 50_000_000)
            for candidate in range(start, end, 2):
                if time.time() > deadline:
                    break
                if candidate % 2 == 0:
                    continue
                if not _is_prime_mr(candidate):
                    continue
                k = quick_smooth_run(candidate, 200)
                if k > best_k:
                    if verify_run(candidate, k):
                        best_k = k
                        best_p = candidate

    # Phase 3: Random sampling of larger primes
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline:
                break
            p = random.randint(10**6, 10**8)
            if p % 2 == 0:
                p += 1
            while not _is_prime_mr(p):
                p += 2
            k = quick_smooth_run(p, 200)
            if k > best_k:
                if verify_run(p, k):
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}
