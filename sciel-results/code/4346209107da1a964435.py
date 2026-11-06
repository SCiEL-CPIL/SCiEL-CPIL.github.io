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
    for attempt in range(20):
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
        if g != n and g > 1:
            return g
    # Fallback: try small factor
    for f in range(5, 1000, 2):
        if n % f == 0:
            return f
    return n  # should not happen

# Precomputed primes for trial division
_SMOOTH_PRIMES_CACHE = {}

def _get_primes_upto(limit):
    """Get primes up to limit, cached."""
    if limit in _SMOOTH_PRIMES_CACHE:
        return _SMOOTH_PRIMES_CACHE[limit]
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            start = i * i
            sieve[start::i] = bytearray(len(sieve[start::i]))
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    _SMOOTH_PRIMES_CACHE[limit] = primes
    return primes

def _gen_primes_upto(limit):
    """Sieve of Eratosthenes to generate primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            start = i * i
            sieve[start::i] = bytearray(len(sieve[start::i]))
    return [i for i, is_p in enumerate(sieve) if is_p]

def is_p_smooth(n, p):
    """
    Check if all prime factors of n are <= p.
    Optimized: quick trial division, then primality test on remainder.
    """
    if n <= 1:
        return True
    if n <= p:
        return True
    
    # Trial division by small primes up to min(p, 5000)
    trial_limit = min(p, 5000)
    primes = _get_primes_upto(trial_limit)
    
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
    
    # n > p and n > trial_limit. Check if n is prime.
    if is_prime(n):
        return False
    
    # n is composite and > p. Use Pollard-Brent.
    d = _pollard_brent(n)
    if d == n:
        # Failed to factor, assume it has a large prime factor
        return False
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

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

def _segmented_sieve(start, end):
    """Yield primes in [start, end] using segmented sieve."""
    if end < 2:
        return []
    limit = int(math.isqrt(end)) + 1
    small_primes = _gen_primes_upto(limit)
    
    size = end - start + 1
    is_prime_arr = bytearray(b'\x01') * size
    is_prime_arr[0] = 0
    
    for p in small_primes:
        # Find first multiple of p in [start, end]
        first = ((start + p - 1) // p) * p
        if first == p:
            first += p
        for j in range(first - start, size, p):
            is_prime_arr[j] = 0
    
    result = []
    for i in range(size):
        if is_prime_arr[i]:
            result.append(start + i)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Precompute small primes
    small_primes = _gen_primes_upto(10000)
    
    # Phase 1: Exhaustive search for small primes up to 2,000,000
    # For small p, the smoothness check is fast
    PRIME_LIMIT_SMALL = 2_000_000
    primes_small = _gen_primes_upto(PRIME_LIMIT_SMALL)
    
    for p in primes_small:
        if time.time() > deadline:
            break
        # Quick filter: check p^2+1 first (most common early failure)
        val1 = p * p + 1
        if val1 > p and not is_p_smooth(val1, p):
            continue
        k = _check_run(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Segmented sieve for [2M, 20M]
    SEG_SIZE = 1_000_000
    seg_start = PRIME_LIMIT_SMALL + 1
    if seg_start % 2 == 0:
        seg_start += 1
    SEGMENT_END = 20_000_000
    
    while seg_start <= SEGMENT_END and time.time() < deadline:
        seg_end = min(seg_start + SEG_SIZE - 1, SEGMENT_END)
        primes_in_seg = _segmented_sieve(seg_start, seg_end)
        for p in primes_in_seg:
            if time.time() > deadline:
                break
            # Quick filter: check p^2+1 first
            val1 = p * p + 1
            if val1 > p and not is_p_smooth(val1, p):
                continue
            k = _check_run(p, 200)
            if k > best_k:
                best_k = k
                best_p = p
        seg_start = seg_end + 1
    
    # Phase 3: Random search for larger primes [20M, 100M]
    # Focus on primes where p^2+1 might be smooth
    if time.time() < deadline - 5:
        # For larger p, p^2+1 is ~14-18 digits. Most will have a prime factor > p.
        # We can use the property that factors of p^2+1 are 1 mod 4 (or 2).
        # Try random odd numbers, filter by small primes, check primality, then smoothness.
        small_filter = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        attempts = 0
        while time.time() < deadline - 3 and attempts < 50000:
            p_candidate = random.randint(20_000_000, 100_000_000)
            if p_candidate % 2 == 0:
                p_candidate += 1
            # Quick filter by small primes
            skip = False
            for q in small_filter:
                if p_candidate % q == 0 and p_candidate != q:
                    skip = True
                    break
            if skip:
                attempts += 1
                continue
            attempts += 1
            if is_prime(p_candidate):
                val1 = p_candidate * p_candidate + 1
                if val1 > p_candidate and not is_p_smooth(val1, p_candidate):
                    continue
                k = _check_run(p_candidate, 200)
                if k > best_k:
                    best_k = k
                    best_p = p_candidate
    
    # Phase 4: Try a few more random primes in [100M, 1e9] if time allows
    if time.time() < deadline - 3:
        small_filter = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        attempts = 0
        while time.time() < deadline - 1 and attempts < 20000:
            p_candidate = random.randint(100_000_000, 999_999_937)
            if p_candidate % 2 == 0:
                p_candidate += 1
            skip = False
            for q in small_filter:
                if p_candidate % q == 0 and p_candidate != q:
                    skip = True
                    break
            if skip:
                attempts += 1
                continue
            attempts += 1
            if is_prime(p_candidate):
                val1 = p_candidate * p_candidate + 1
                if val1 > p_candidate and not is_p_smooth(val1, p_candidate):
                    continue
                k = _check_run(p_candidate, 200)
                if k > best_k:
                    best_k = k
                    best_p = p_candidate
    
    return {'p': best_p, 'k': best_k}
