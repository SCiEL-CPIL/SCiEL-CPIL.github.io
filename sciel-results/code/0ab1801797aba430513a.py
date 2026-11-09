import time
import math
import random
import numpy as np

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
    if _is_prime_mr(t):
        return t <= p
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
    if _is_prime_mr(t):
        return t <= p
    factors = []
    _factor(t, factors)
    return all(f <= p for f in factors)

def _batch_smoothness_check(p_sq, k, p, small_primes, small_limit):
    """
    Vectorized smoothness check for p^2+1 ... p^2+k.
    Returns the maximum run length of p-smooth numbers starting from p^2+1.
    Uses NumPy to divide out small primes simultaneously, then checks remainders.
    """
    # Create array of values p^2+1, ..., p^2+k
    vals = p_sq + np.arange(1, k + 1, dtype=np.int64)
    
    # Track which values are still "alive" (not yet proven to have a factor > p)
    alive = np.ones(k, dtype=bool)
    
    # Trial divide by small primes up to min(p, small_limit)
    for q in small_primes:
        if q > p:
            break
        if q > small_limit:
            break
        # For alive values, divide out all factors of q
        mask = alive & (vals % q == 0)
        if not np.any(mask):
            continue
        # Divide out q repeatedly from masked values
        while np.any(mask):
            vals[mask] //= q
            mask = alive & (vals % q == 0)
    
    # Now check each value: is the remaining factor <= p?
    # For values where vals[i] == 1, they're fully factored -> smooth
    # For vals[i] > 1, need to check if all prime factors <= p
    # Quick check: if vals[i] is prime and vals[i] <= p, smooth
    # If vals[i] is prime and vals[i] > p, NOT smooth
    # If vals[i] is composite, need full factorization
    
    # Find the longest prefix where all values are p-smooth
    max_run = 0
    for i in range(k):
        if not alive[i]:
            break
        v = int(vals[i])
        if v == 1:
            max_run = i + 1
            continue
        # Check if v is prime
        if _is_prime_mr(v):
            if v <= p:
                max_run = i + 1
            else:
                break
        else:
            # v is composite; factor it
            factors = []
            _factor(v, factors)
            if all(f <= p for f in factors):
                max_run = i + 1
            else:
                break
    
    return max_run

def _batch_smoothness_check_v2(p_sq, k, p, small_primes, small_limit):
    """
    Optimized vectorized smoothness check.
    Key insight: for p^2 + i where i is small (1..200), the values are close to p^2.
    Any prime factor q of p^2+i with q <= p means p^2 ≡ -i (mod q), i.e., p ≡ ±sqrt(-i) (mod q).
    We precompute for each small prime q whether p^2 + i is divisible by q.
    
    Strategy: 
    1. For each small prime q <= p, compute which i in [1,k] have q | (p^2+i)
    2. Divide out all small prime factors
    3. Check remaining factors
    """
    vals = p_sq + np.arange(1, k + 1, dtype=np.int64)
    
    # Divide out small primes
    for q in small_primes:
        if q > p:
            break
        if q > small_limit:
            break
        # Find indices where q divides vals
        mask = vals % q == 0
        if not np.any(mask):
            continue
        # Divide out all powers of q
        while np.any(mask):
            vals[mask] //= q
            mask = vals % q == 0
    
    # Check longest prefix of p-smooth numbers
    max_run = 0
    for i in range(k):
        v = int(vals[i])
        if v == 1:
            max_run = i + 1
            continue
        if _is_prime_mr(v):
            if v <= p:
                max_run = i + 1
            else:
                break
        else:
            factors = []
            _factor(v, factors)
            if all(f <= p for f in factors):
                max_run = i + 1
            else:
                break
    
    return max_run

def _generate_primes_in_range(lo, hi, sieve_limit):
    """Generate primes in [lo, hi] using a segmented sieve approach with Miller-Rabin."""
    primes = []
    if lo <= 2 <= hi:
        primes.append(2)
    start = max(3, lo if lo % 2 == 1 else lo + 1)
    for n in range(start, hi + 1, 2):
        if _is_prime_mr(n):
            primes.append(n)
        if len(primes) > 50000:  # safety cap
            break
    return primes

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Precompute small primes up to 100000 for fast trial division
    SIEVE_LIMIT = 100000
    small_primes = _sieve(SIEVE_LIMIT)
    
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

    # Phase 2: Search larger primes using vectorized batch smoothness check
    random.seed(42)
    
    # For larger p, use the batched approach which is much faster per candidate
    # because it processes all 200 values simultaneously with NumPy
    
    # Search ranges for larger primes, increasing in size
    # Use segmented generation for better coverage
    search_ranges = [
        (100000, 300000),
        (300000, 1000000),
        (1000000, 5000000),
        (5000000, 20000000),
        (20000000, 100000000),
        (100000000, 500000000),
        (500000000, 1000000000),
    ]

    for lo, hi in search_ranges:
        if time.time() > deadline:
            break
        time_remaining = deadline - time.time()
        if time_remaining < 2:
            break
        
        # For each range, sample primes and use batched smoothness check
        # The batched check is ~10-50x faster per candidate than sequential
        # so we can test many more candidates
        
        # Estimate: batched check takes ~2-5ms per candidate (mostly NumPy overhead)
        # vs ~20-100ms for sequential. So we can test 5-20x more candidates.
        num_samples = min(5000, int(time_remaining * 200))
        num_samples = max(num_samples, 100)
        
        tested = 0
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
            
            tested += 1
            p_sq = p * p
            
            # Use batched smoothness check
            k = _batch_smoothness_check_v2(p_sq, 200, p, small_primes, SIEVE_LIMIT)
            
            if k > best_k:
                best_k = k
                best_p = p
                if best_k >= 200:
                    return {'p': best_p, 'k': best_k}

    return {'p': best_p, 'k': best_k}
