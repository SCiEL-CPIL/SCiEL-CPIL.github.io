import time
import math
import random
from functools import lru_cache
import sys

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

# ---------- Factorization with cache ----------
_factor_cache = {}

def _factor(n):
    if n in _factor_cache:
        return _factor_cache[n]
    if n == 1:
        _factor_cache[n] = ()
        return ()
    if _is_prime_mr(n):
        _factor_cache[n] = (n,)
        return (n,)
    d = _pollard_brent(n)
    result = _factor(d) + _factor(n // d)
    _factor_cache[n] = result
    return result

def _all_factors_at_most_p(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
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

    # Precompute primes up to 50k for fast sieving. 
    # Larger limits make the sieve loop slower per candidate.
    # We rely on Pollard-Brent for the rest.
    SMALL_LIMIT = 50_000
    small_primes = sieve_primes(SMALL_LIMIT)
    
    # For pre-filter: primes up to 100k
    PREFILTER_LIMIT = 100_000
    prefilter_primes = [q for q in small_primes if q <= PREFILTER_LIMIT]

    def check_run_fast(p, max_k=200):
        """
        Optimized check: Sieve out small primes, then check remainders.
        """
        p_sq = p * p
        segment_size = max_k
        rem = [p_sq + i for i in range(1, segment_size + 1)]
        
        # Sieve: divide out all primes q <= min(p, SMALL_LIMIT)
        # We only sieve up to SMALL_LIMIT to keep it fast.
        # If p > SMALL_LIMIT, we still only sieve up to SMALL_LIMIT.
        # Any remaining factor > SMALL_LIMIT must be checked individually.
        limit_q = min(p, SMALL_LIMIT)
        
        for q in small_primes:
            if q > limit_q:
                break
            p2_mod_q = (p * p) % q
            target = (-p2_mod_q) % q
            if target == 0:
                first = q
            else:
                first = target
            if first > segment_size:
                continue
            for idx in range(first - 1, segment_size, q):
                while rem[idx] % q == 0:
                    rem[idx] //= q
        
        k = 0
        for i in range(1, segment_size + 1):
            r = rem[i - 1]
            if r == 1:
                k += 1
                continue
            # r > 1. r has no prime factors <= limit_q.
            # If p <= limit_q, then r has no prime factors <= p.
            # So if r > 1, it has a factor > p. Break.
            if p <= limit_q:
                break
            # If p > limit_q, r might have factors in (limit_q, p].
            # We need to check if all prime factors of r are <= p.
            # Since r is at most ~p^2, and we removed small factors, r is likely prime or product of 2 large primes.
            # Fast check: Is r prime?
            if _is_prime_mr(r):
                if r <= p:
                    k += 1
                else:
                    break
            else:
                # r is composite. Factor it.
                # If any factor > p, break.
                # Optimization: If r > p*p, it must have a factor > p? No.
                # Just factor and check.
                try:
                    factors = _factor(r)
                    if all(f <= p for f in factors):
                        k += 1
                    else:
                        break
                except:
                    break
        return k

    def check_run_full(p, max_k=200):
        return check_run_fast(p, max_k)

    def prefilter(p):
        """
        Quick rejection: check if any prime q in (p, PREFILTER_LIMIT] divides p^2+i for some i in [1,200].
        If so, that n has a factor > p, so the run is broken at that point.
        Returns the minimum i in [1,200] where p^2+i is divisible by some q in (p, PREFILTER_LIMIT],
        or 201 if no such collision found.
        """
        min_bad = 201
        for q in prefilter_primes:
            if q <= p:
                continue
            # Check if p^2 + i ≡ 0 (mod q) for some i in [1, 200]
            # i ≡ -p^2 (mod q)
            p2_mod_q = (p * p) % q
            target = (-p2_mod_q) % q
            if target == 0:
                i = q
            else:
                i = target
            if 1 <= i <= 200:
                if i < min_bad:
                    min_bad = i
        return min_bad

    def verify_run(p, k):
        """Full verification that all of p^2+1..p^2+k are p-smooth."""
        p_sq = p * p
        for i in range(1, k + 1):
            n = p_sq + i
            if not _all_factors_at_most_p(n, p):
                return False
        return True

    # Phase 1: Search small primes up to 500k (small_primes covers 50k, so we need more primes for search)
    # Let's generate primes up to 1M for search, but use SMALL_LIMIT=50k for sieving.
    search_primes = sieve_primes(1_000_000)
    
    for p in search_primes:
        if time.time() > deadline:
            break
        # Pre-filter: if any q in (p, PREFILTER_LIMIT] hits, the run is at most min_bad-1
        # Note: prefilter_primes only goes up to 100k. If p > 100k, prefilter is weak.
        pf = prefilter(p)
        max_possible = min(200, pf - 1)
        if max_possible <= best_k:
            continue
        k = check_run_full(p, max_possible)
        if k > best_k:
            if verify_run(p, k):
                best_k = k
                best_p = p

    # Phase 2: Random search for larger primes
    # Long runs are rare for large p. Random sampling is better than exhaustive.
    if time.time() < deadline - 10:
        for _ in range(5000):
            if time.time() > deadline:
                break
            # Sample primes up to 10^7
            p = random.randint(10**6, 10**7)
            if p % 2 == 0:
                p += 1
            while not _is_prime_mr(p):
                p += 2
            pf = prefilter(p)
            max_possible = min(200, pf - 1)
            if max_possible <= best_k:
                continue
            k = check_run_full(p, max_possible)
            if k > best_k:
                if verify_run(p, k):
                    best_k = k
                    best_p = p

    # Phase 3: Final check with known small primes if time remains
    # Sometimes small primes have surprisingly long runs
    if time.time() < deadline - 2:
        for p in search_primes[:100]:
            if time.time() > deadline:
                break
            pf = prefilter(p)
            max_possible = min(200, pf - 1)
            if max_possible <= best_k:
                continue
            k = check_run_full(p, max_possible)
            if k > best_k:
                if verify_run(p, k):
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}
