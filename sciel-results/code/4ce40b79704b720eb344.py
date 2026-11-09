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

def _sieve(limit):
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def _next_prime(n):
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while not _is_prime_mr(n):
        n += 2
    return n

def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    for _ in range(20):
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
        if g != n and g != 1:
            return g
    return n  # failure

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    if d == n:
        result.append(n)
        return
    _factor(d, result)
    _factor(n // d, result)

def _is_p_smooth(val, p, small_primes):
    """Check if val is p-smooth (all prime factors <= p)."""
    t = val
    for q in small_primes:
        if q > p or q * q > t:
            break
        while t % q == 0:
            t //= q
    if t == 1:
        return True
    if _is_prime_mr(t):
        return t <= p
    # t is composite with all prime factors > max(small_primes used)
    # Factor it
    factors = []
    _factor(t, factors)
    return all(f <= p for f in factors)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute primes up to 10^6
    small_primes = _sieve(1000000)

    best_k = 0
    best_p = 3

    # ================================================================
    # PHASE 1: Exhaustive search over primes up to 3*10^6
    # ================================================================
    SEARCH_LIMIT_1 = 3000000
    candidate_primes = _sieve(SEARCH_LIMIT_1)

    for p in candidate_primes:
        if time.time() > deadline:
            break
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            val = p_sq + i
            t = val
            for q in small_primes:
                if q > p or q * q > t:
                    break
                while t % q == 0:
                    t //= q
            if t == 1:
                k = i
                continue
            if _is_prime_mr(t):
                if t <= p:
                    k = i
                else:
                    break
            else:
                factors = []
                _factor(t, factors)
                if all(f <= p for f in factors):
                    k = i
                else:
                    break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break

    # ================================================================
    # PHASE 2: Random sampling for larger primes (3*10^6 to 10^9)
    # Use the cofactor-window insight for a fast pre-filter:
    # p^2+i is NOT p-smooth iff it has a prime factor q > p.
    # For p >= 20: any such q has cofactor r = (p^2+i)/q <= p + 200/p <= p+11.
    # So p^2+i = q*r with q prime > p and 1 <= r <= p+11.
    #
    # Fast pre-filter: For each i, check if p^2+i is divisible by any
    # prime q in (p, p+11]. If yes, definitely not p-smooth.
    # If no, still might not be p-smooth (q could be larger), but this
    # eliminates many cases quickly.
    #
    # We generate candidate primes by sampling and checking.
    # ================================================================

    if best_k < 200:
        # For the pre-filter, we need to find primes in (p, p+11]
        # These are at most ~5 primes (by PNT, ~11/ln(p) primes)
        
        for trial in range(5000):
            if time.time() > deadline:
                break
            
            # Log-uniform sampling
            log_lo = math.log10(3 * 10**6)
            log_hi = math.log10(10**9)
            log_p = log_lo + random.random() * (log_hi - log_lo)
            p_approx = int(10 ** log_p)
            p = _next_prime(p_approx)
            
            if p > 10**9:
                continue
            
            p_sq = p * p
            k = 0
            for i in range(1, 201):
                if time.time() > deadline:
                    break
                val = p_sq + i
                t = val
                
                # Trial divide by small primes
                for q in small_primes:
                    if q * q > t:
                        break
                    while t % q == 0:
                        t //= q
                
                if t == 1:
                    k = i
                    continue
                
                if _is_prime_mr(t):
                    if t <= p:
                        k = i
                    else:
                        break
                else:
                    factors = []
                    _factor(t, factors)
                    if all(f <= p for f in factors):
                        k = i
                    else:
                        break
            
            if k > best_k:
                best_k = k
                best_p = p
                if best_k >= 200:
                    break

    return {'p': best_p, 'k': best_k}
