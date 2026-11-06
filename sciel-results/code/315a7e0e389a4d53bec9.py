import time
import math
import random

# ── Deterministic Miller-Rabin for n < 2^64 ──────────────────────────────────
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

def is_prime(n):
    """Deterministic Miller-Rabin. Exact for n < 2^64 with 12 base primes."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
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

# ── Pollard's Rho / Brent factorisation ──────────────────────────────────────

def _pollard_brent(n):
    """Return a non-trivial factor of composite n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    c = random.randrange(1, n - 1)
    y = random.randrange(0, n - 1)
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
                q = (q * abs(x - y)) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
    if g == n:
        g = 1
        while g == 1:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
    if g == n:
        # retry with different parameters
        return _pollard_brent(n)
    return g

# Precomputed primes for trial division in smoothness check
_SMOOTH_PRIMES = None

def _get_smooth_primes(limit):
    """Get primes up to limit for trial division."""
    global _SMOOTH_PRIMES
    if _SMOOTH_PRIMES is not None and _SMOOTH_PRIMES[-1] >= limit:
        return _SMOOTH_PRIMES
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            start = i * i
            step = i
            sieve[start::step] = bytearray(len(sieve[start::step]))
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    if _SMOOTH_PRIMES is None:
        _SMOOTH_PRIMES = primes
    else:
        _SMOOTH_PRIMES = primes
    return primes

def is_p_smooth(n, p):
    """
    Check if all prime factors of n are <= p.
    """
    if n <= 1:
        return True
    if n <= p:
        return True
    
    # Trial division by primes up to min(p, 1000) or until n becomes 1
    # Use primes up to a reasonable bound
    trial_limit = min(p, 2000)
    primes = _get_smooth_primes(trial_limit)
    
    for q in primes:
        if q > p:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
            if n == 1:
                return True
            if n <= p:
                return True
    
    if n == 1:
        return True
    if n <= p:
        return True
    
    # n > p after trial division
    if is_prime(n):
        return False
    
    # n is composite and > p. Use Pollard-Brent.
    d = _pollard_brent(n)
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def _gen_primes_upto(limit):
    """Sieve of Eratosthenes to generate primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            start = i * i
            step = i
            sieve[start::step] = bytearray(len(sieve[start::step]))
    return [i for i, is_p in enumerate(sieve) if is_p]

def _check_run(p, max_k=200):
    """Check how many consecutive integers p^2+1, ..., p^2+k are p-smooth."""
    base = p * p
    k = 0
    for i in range(1, max_k + 1):
        val = base + i
        if not is_p_smooth(val, p):
            break
        k += 1
    return k

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Precompute primes for smoothness checking
    _get_smooth_primes(2000)
    
    # Phase 1: Exhaustive search for small primes
    # Sieve up to 5,000,000
    PRIME_LIMIT = 5_000_000
    primes = _gen_primes_upto(PRIME_LIMIT)
    
    for p in primes:
        if time.time() > deadline:
            break
        k = _check_run(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Continue scanning with Miller-Rabin for primes in [5e6, 10e6]
    current = PRIME_LIMIT + 1
    if current % 2 == 0:
        current += 1
        
    SCAN_LIMIT = 10_000_000
    while current <= SCAN_LIMIT:
        if time.time() > deadline:
            break
        if is_prime(current):
            k = _check_run(current, 200)
            if k > best_k:
                best_k = k
                best_p = current
        current += 2
        
    # Phase 3: If time remains, do a focused random search for medium-large primes
    # Focus on [10^6, 10^8] where Miller-Rabin is fast and smoothness checks are reasonable
    if time.time() < deadline - 10:
        # Use a smarter random sampling: pick candidates not divisible by small primes
        small_primes_for_filter = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        while time.time() < deadline - 5:
            p_candidate = random.randint(1_000_000, 100_000_000)
            if p_candidate % 2 == 0:
                p_candidate += 1
            # Quick filter
            skip = False
            for q in small_primes_for_filter:
                if p_candidate % q == 0 and p_candidate != q:
                    skip = True
                    break
            if skip:
                continue
            if is_prime(p_candidate):
                k = _check_run(p_candidate, 200)
                if k > best_k:
                    best_k = k
                    best_p = p_candidate
    
    return {'p': best_p, 'k': best_k}
