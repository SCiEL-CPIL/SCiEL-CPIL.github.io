import time
import math
import random

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

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

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, result)
    _factor(n // d, result)

# ---------- Sieve ----------
def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def _is_p_smooth_trial(n, p, small_primes):
    """Check if n is p-smooth using trial division by small_primes up to p."""
    t = n
    for q in small_primes:
        if q > p:
            break
        if q * q > t:
            break
        while t % q == 0:
            t //= q
    if t == 1:
        return True
    # t is either prime or composite with all factors > smallest prime not tried
    if _is_prime_mr(t):
        return t <= p
    # t is composite; factor it
    factors = []
    _factor(t, factors)
    return all(f <= p for f in factors)

def _is_p_smooth_hybrid(n, p, small_primes, small_limit):
    """Check if n is p-smooth: trial divide by small_primes, then Pollard-Brent on remainder."""
    t = n
    for q in small_primes:
        if q * q > t:
            break
        while t % q == 0:
            t //= q
    if t == 1:
        return True
    # Now t has no factors <= small_limit
    # If t is prime, check if t <= p
    if _is_prime_mr(t):
        return t <= p
    # t is composite with all prime factors > small_limit
    # Factor t and check all factors <= p
    factors = []
    _factor(t, factors)
    return all(f <= p for f in factors)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Precompute small primes up to 100000 for fast trial division
    SIEVE_LIMIT = 100000
    small_primes = _sieve(SIEVE_LIMIT)
    small_prime_set = set(small_primes)

    best_k = 0
    best_p = 3

    # Phase 1: Search small primes (up to 100000) with full trial division
    for p in small_primes:
        if time.time() > deadline:
            break
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            val = p_sq + i
            if _is_p_smooth_trial(val, p, small_primes):
                k = i
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                return {'p': best_p, 'k': best_k}

    # Phase 2: Search larger primes using hybrid approach
    # For p > 100000, trial division up to p is too slow; use hybrid
    random.seed(42)
    
    # Search ranges for larger primes
    search_ranges = [
        (100000, 500000),
        (500000, 2000000),
        (2000000, 10000000),
        (10000000, 100000000),
        (100000000, 1000000000),
    ]

    for lo, hi in search_ranges:
        if time.time() > deadline:
            break
        # Sample primes in this range
        # Number of samples based on time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 2:
            break
        
        # Estimate: each prime check takes ~0.5ms for small k, up to ~50ms for k=200
        # Be conservative: sample at most 2000 per range
        num_samples = min(2000, int(time_remaining * 50))
        num_samples = max(num_samples, 100)
        
        for _ in range(num_samples):
            if time.time() > deadline:
                break
            # Generate random odd number in range
            p = random.randint(lo, hi)
            if p % 2 == 0:
                p += 1
            if p < 2 or p > 10**9:
                continue
            
            # Find next prime
            attempts = 0
            while not _is_prime_mr(p) and attempts < 200:
                p += 2
                attempts += 1
            if p > 10**9 or not _is_prime_mr(p):
                continue
            
            p_sq = p * p
            k = 0
            for i in range(1, 201):
                if time.time() > deadline:
                    break
                val = p_sq + i
                if _is_p_smooth_hybrid(val, p, small_primes, SIEVE_LIMIT):
                    k = i
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
                if best_k >= 200:
                    return {'p': best_p, 'k': best_k}

    return {'p': best_p, 'k': best_k}
