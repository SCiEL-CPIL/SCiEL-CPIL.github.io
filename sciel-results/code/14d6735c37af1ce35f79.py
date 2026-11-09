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

    # Precompute primes up to 200k for fast sieving and collision detection
    SIEVE_LIMIT = 200_000
    all_primes = sieve_primes(SIEVE_LIMIT)
    prime_set = set(all_primes)
    
    # For collision detection: primes up to 500k
    COLLISION_LIMIT = 500_000
    collision_primes = sieve_primes(COLLISION_LIMIT)
    
    # Precompute small primes for fast trial division
    SMALL_PRIMES = [q for q in all_primes if q <= 1000]

    def check_run_fast(p, max_k=200):
        """
        Optimized check: Sieve out small primes, then check remainders.
        """
        p_sq = p * p
        segment_size = max_k
        rem = [p_sq + i for i in range(1, segment_size + 1)]
        
        # Sieve: divide out all primes q <= min(p, SIEVE_LIMIT)
        limit_q = min(p, SIEVE_LIMIT)
        
        for q in all_primes:
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
            if p <= limit_q:
                break
            # If p > limit_q, r might have factors in (limit_q, p].
            if _is_prime_mr(r):
                if r <= p:
                    k += 1
                else:
                    break
            else:
                try:
                    factors = _factor(r)
                    if all(f <= p for f in factors):
                        k += 1
                    else:
                        break
                except:
                    break
        return k

    def collision_filter(p, max_k=200):
        """
        Bitset-based collision detection: for each prime q in (p, COLLISION_LIMIT],
        mark which positions i in [1, max_k] are hit by q (i.e., q | p²+i).
        Return the minimum i not marked (or max_k+1 if all are marked).
        """
        # Use a bytearray as a bitset for positions 1..max_k
        hit = bytearray(max_k + 1)  # hit[i] = 1 if some q > p divides p²+i
        
        for q in collision_primes:
            if q <= p:
                continue
            if q > COLLISION_LIMIT:
                break
            # i ≡ -p² (mod q)
            p2_mod_q = (p * p) % q
            target = (-p2_mod_q) % q
            if target == 0:
                first = q
            else:
                first = target
            if first > max_k:
                continue
            for idx in range(first - 1, max_k, q):
                hit[idx] = 1
        
        # Find the first unmarked position
        for i in range(1, max_k + 1):
            if not hit[i]:
                return i
        return max_k + 1

    def verify_run(p, k):
        """Full verification that all of p^2+1..p^2+k are p-smooth."""
        p_sq = p * p
        for i in range(1, k + 1):
            n = p_sq + i
            if not _all_factors_at_most_p(n, p):
                return False
        return True

    # Phase 1: Exhaustive search of small primes up to 1M
    # For small p, the collision filter is very effective because there are many primes q in (p, COLLISION_LIMIT]
    search_primes_small = [p for p in all_primes if p <= 1_000_000]
    
    for p in search_primes_small:
        if time.time() > deadline:
            break
        # Quick collision check: if any q in (p, COLLISION_LIMIT] hits position 1, run is 0
        # Use a fast pre-filter: check if p²+1 has a factor > p
        n1 = p * p + 1
        if _is_prime_mr(n1) and n1 > p:
            continue  # p²+1 is prime > p, so run is 0
        # Check collision for first few positions
        first_bad = collision_filter(p, min(200, best_k + 1))
        if first_bad <= best_k:
            continue
        # Full check
        k = check_run_fast(p, 200)
        if k > best_k:
            if verify_run(p, k):
                best_k = k
                best_p = p

    # Phase 2: Random search for larger primes with collision filtering
    # For large p, use random sampling but with efficient collision filtering
    if time.time() < deadline - 10:
        attempts = 0
        while time.time() < deadline - 5 and attempts < 20000:
            attempts += 1
            # Sample primes in [10^6, 10^8]
            p = random.randint(10**6, 10**8)
            if p % 2 == 0:
                p += 1
            # Find next prime
            while not _is_prime_mr(p):
                p += 2
                if p > 10**8:
                    break
            if p > 10**8:
                continue
            
            # Quick rejection: check if p²+1 has a factor > p
            n1 = p * p + 1
            # Trial divide by small primes
            has_large_factor = False
            temp = n1
            for q in SMALL_PRIMES:
                if q * q > temp:
                    break
                while temp % q == 0:
                    temp //= q
            if temp > 1:
                if _is_prime_mr(temp):
                    if temp > p:
                        has_large_factor = True
                else:
                    # Composite, check if all factors <= p
                    try:
                        factors = _factor(temp)
                        if any(f > p for f in factors):
                            has_large_factor = True
                    except:
                        has_large_factor = True
            if has_large_factor:
                continue
            
            # Collision filter for first best_k+1 positions
            first_bad = collision_filter(p, min(200, best_k + 1))
            if first_bad <= best_k:
                continue
            
            # Full check
            k = check_run_fast(p, 200)
            if k > best_k:
                if verify_run(p, k):
                    best_k = k
                    best_p = p

    # Phase 3: Targeted search around known good ranges
    # Sometimes clusters of good primes exist
    if time.time() < deadline - 2:
        # Try primes near 10^5, 10^6, etc.
        for base in [10**4, 5*10**4, 10**5, 5*10**5, 10**6, 5*10**6]:
            if time.time() > deadline:
                break
            # Sample 100 primes in [base, base*2]
            for _ in range(100):
                if time.time() > deadline:
                    break
                p = random.randint(base, min(base * 2, 10**8))
                if p % 2 == 0:
                    p += 1
                while not _is_prime_mr(p):
                    p += 2
                if p > 10**8:
                    continue
                
                # Quick rejection
                n1 = p * p + 1
                temp = n1
                for q in SMALL_PRIMES:
                    if q * q > temp:
                        break
                    while temp % q == 0:
                        temp //= q
                if temp > 1:
                    if _is_prime_mr(temp):
                        if temp > p:
                            continue
                    else:
                        try:
                            factors = _factor(temp)
                            if any(f > p for f in factors):
                                continue
                        except:
                            continue
                
                first_bad = collision_filter(p, min(200, best_k + 1))
                if first_bad <= best_k:
                    continue
                
                k = check_run_fast(p, 200)
                if k > best_k:
                    if verify_run(p, k):
                        best_k = k
                        best_p = p

    return {'p': best_p, 'k': best_k}
