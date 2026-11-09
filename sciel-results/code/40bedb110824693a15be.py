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

def sieve_primes(limit):
    """Generate list of all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def is_p_smooth(n, p, small_primes):
    """Check if all prime factors of n are <= p.
    Uses trial division by small primes, then Miller-Rabin, then Pollard-Brent for remainder."""
    if n <= 1:
        return True
    if is_prime(n):
        return n <= p

    v = n
    sqrt_n = int(math.isqrt(n))
    limit = min(p, sqrt_n)

    for sp in small_primes:
        if sp > limit:
            break
        while v % sp == 0:
            v //= sp
        if v == 1:
            return True

    if v == 1:
        return True

    # v is not 1; all prime factors of v are > last trial prime used
    if is_prime(v):
        return v <= p

    # v is composite; factor it with Pollard rho and check both parts
    d = pollard_rho(v)
    if d > p:
        return False
    if v // d > p:
        return False
    # Both d and v//d are <= p; recursively verify (depth limited by factor count)
    return is_p_smooth(d, p, small_primes) and is_p_smooth(v // d, p, small_primes)

def check_run_length(base, p, small_primes, max_k=200):
    """Count consecutive p-smooth numbers starting from base+1."""
    k = 0
    for i in range(1, max_k + 1):
        if is_p_smooth(base + i, p, small_primes):
            k += 1
        else:
            break
    return k

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill

    best_k = 0
    best_p = 2

    # Precompute small primes for trial division - up to 10^6
    SIEVE_LIMIT = 1000000
    small_primes = sieve_primes(SIEVE_LIMIT)

    # Phase 1: Exhaustive search over primes up to 2,000,000
    search_limit = 2000000
    search_primes = sieve_primes(search_limit)

    for p in search_primes:
        if time.time() > deadline:
            break
        if p < 2:
            continue
        base = p * p
        k = check_run_length(base, p, small_primes)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Segmented sieve for larger primes
    if best_k < 200 and time.time() < deadline - 30:
        for start, end in [(search_limit, 20000000), (20000000, 100000000)]:
            if time.time() > deadline - 20:
                break
            window_size = 500000
            pos = start
            while pos < end and time.time() < deadline - 20:
                sieve_end = min(pos + window_size, end)
                sieve = bytearray(b'\x01') * (sieve_end - pos)
                for sp in small_primes:
                    if sp * sp > sieve_end:
                        break
                    first = ((pos + sp - 1) // sp) * sp
                    if first < sp * sp:
                        first = sp * sp
                    for j in range(first - pos, sieve_end - pos, sp):
                        sieve[j] = 0
                for i in range(sieve_end - pos):
                    if time.time() > deadline - 20:
                        break
                    cand = pos + i
                    if cand < 2:
                        continue
                    if sieve[i] and is_prime(cand):
                        base = cand * cand
                        k = check_run_length(base, cand, small_primes)
                        if k > best_k:
                            best_k = k
                            best_p = cand
                pos = sieve_end

    if best_p == 0:
        best_p = 2
        best_k = 1

    return {'p': best_p, 'k': best_k}
