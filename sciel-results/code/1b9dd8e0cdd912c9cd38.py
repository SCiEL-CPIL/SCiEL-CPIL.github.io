import time
import math
import random

# Deterministic Miller-Rabin for 64-bit integers
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    # Small prime check
    small_primes = (5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for sp in small_primes:
        if n % sp == 0:
            return n == sp
    # Miller-Rabin
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
            x = (x * x) % n
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
    if n % 5 == 0:
        return 5
    while True:
        c = random.randrange(1, n)
        x = random.randrange(2, n - 1)
        y = x
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

_SMALL_TRIAL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)

def factorize(n):
    """Return list of prime factors of n (with multiplicity)."""
    if n <= 1:
        return []
    if is_prime(n):
        return [n]
    factors = []
    for sp in _SMALL_TRIAL:
        if n % sp == 0:
            while n % sp == 0:
                factors.append(sp)
                n //= sp
            if n == 1:
                return factors
            if is_prime(n):
                factors.append(n)
                return factors
    d = pollard_rho(n)
    return factorize(d) + factorize(n // d)

def _sieve(limit):
    """Return list of primes up to limit using bytearray sieve."""
    if limit < 2:
        return []
    flags = bytearray([1]) * (limit + 1)
    flags[0] = flags[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if flags[i]:
            flags[i * i: limit + 1: i] = bytearray(len(range(i * i, limit + 1, i)))
    return [i for i in range(2, limit + 1) if flags[i]]

def _is_p_smooth(n, p):
    """Check if all prime factors of n are <= p. Returns True if so."""
    if n <= 1:
        return True
    if is_prime(n):
        return n <= p
    for sp in _SMALL_TRIAL:
        if n % sp == 0:
            while n % sp == 0:
                n //= sp
            if n == 1:
                return True
            if is_prime(n):
                return n <= p
    d = pollard_rho(n)
    return _is_p_smooth(d, p) and _is_p_smooth(n // d, p)

def _run_length(p, max_k=200):
    """Compute the run length starting at p^2+1."""
    p_sq = p * p
    k = 0
    for i in range(1, max_k + 1):
        n = p_sq + i
        if not _is_p_smooth(n, p):
            break
        k += 1
    return k

def construct():
    t0 = time.time()
    deadline = t0 + 115  # 35s safety margin under 150s limit

    best_k = 0
    best_p = 2

    # Phase 1: Sieve and check all primes up to 20 million
    # 20M sieve takes ~0.5s, checking ~1.27M primes takes the rest
    limit1 = 20_000_000
    primes = _sieve(limit1)

    for p in primes:
        if time.time() > deadline:
            break
        k = _run_length(p)
        if k > best_k:
            best_k = k
            best_p = p
        # Early exit: if we've already found k=200, we can't do better
        if k >= 200:
            break

    # Phase 2: If time remains, check larger primes in segments
    # We'll check primes in ranges [20M, 50M], [50M, 100M], etc.
    # Use a segmented approach: generate primes in a window, check them.
    # For larger p, p^2 is up to 10^18, factorization is slower.
    # But we only need to check a few to see if the run breaks quickly.
    if time.time() < deadline - 10:
        # Check primes in [20M, 80M] — about 3.5M more primes
        # Use a simple trial-division primality test with the sieve primes
        # up to sqrt(80M) ≈ 8944
        sieve_small = _sieve(10_000)  # primes up to 10000 for trial division
        lo = limit1 + 1
        hi = 80_000_000
        # We'll scan in chunks
        chunk_size = 1_000_000
        for start in range(lo, hi, chunk_size):
            if time.time() > deadline:
                break
            end = min(start + chunk_size, hi)
            # Generate candidate odd numbers in [start, end]
            # Check primality via trial division with sieve_small
            # This is slow, so only do a few chunks
            # Actually, for large p the run is almost certainly short (1-2),
            # so the expected gain is low. Let's skip this phase unless
            # we have a lot of time left.
            pass

    return {'p': best_p, 'k': best_k}
