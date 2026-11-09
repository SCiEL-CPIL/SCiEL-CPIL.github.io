"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.3e24."""
    if n < 2:
        return False
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        return True
    if n % 2 == 0:
        return False
    # write n-1 = d*2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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

def _pollard_brent(n):
    """Pollard's Brent cycle detection for finding a factor of n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(2, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, n - 1)
        g = r = q = 1
        x = ys = y
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
                if g > 1:
                    break
        if g != n:
            return g

def _factor(n, primes=None):
    """Full factorization of n. Returns dict {prime: exponent}."""
    if n <= 1:
        return {}
    result = {}
    # trial division by small primes
    if primes:
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                cnt = 0
                while n % p == 0:
                    n //= p
                    cnt += 1
                result[p] = cnt
    if n > 1:
        if _is_prime_mr(n):
            result[n] = result.get(n, 0) + 1
        else:
            # use Pollard-Brent
            stack = [n]
            while stack:
                m = stack.pop()
                if m == 1:
                    continue
                if _is_prime_mr(m):
                    result[m] = result.get(m, 0) + 1
                else:
                    d = _pollard_brent(m)
                    stack.append(d)
                    stack.append(m // d)
    return result

def _sieve(limit):
    """Sieve of Eratosthenes, returns list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    # Precompute small primes for trial division
    SIEVE_LIMIT = 100000
    small_primes = _sieve(SIEVE_LIMIT)
    
    # For the vectorized sieve, we only need primes up to sqrt(max_n)
    # If we search p up to 10^7, p^2 is 10^14, sqrt is 10^7.
    # But we can't sieve to 10^7 quickly enough in Python? 
    # Sieve to 10^6 is fast. Let's use primes up to 10000 for the fast filter.
    # Any number with a factor > 10000 will be handled by the slow path.
    # Actually, if we want to find the *largest* prime factor, we need to divide out all small ones.
    
    # Let's optimize the vectorized check.
    # We check p^2 + 1 ... p^2 + 200.
    # We want to know if max_prime_factor(n) <= p for all n.
    
    # Strategy:
    # 1. Generate candidate primes p.
    # 2. For each p, create array N = p^2 + [1..200].
    # 3. For each small prime q <= sqrt(max(N)), if q <= p, divide N by q as much as possible.
    #    If q > p, we don't care about removing it? 
    #    Wait. The condition is "all prime factors <= p".
    #    So if a number has a factor q > p, it fails.
    #    If we divide out all factors q <= p, the remainder R must be 1.
    #    If R > 1, then R has a prime factor > p (since all <= p were removed).
    #    So the check is: after removing all prime factors q <= p, is the remainder 1?
    
    # Optimization:
    # We can precompute the primes to use for division.
    # For a given p, we divide by all primes q <= p.
    # This is expensive if p is large (many primes).
    # However, for p up to 10^6, there are ~78k primes.
    # Dividing an array of 200 elements by 78k primes is 200 * 78k = 15.6M operations.
    # In NumPy, this is very fast (vectorized).
    
    # Let's refine the search strategy.
    # Phase 1: Small p (p < 1000). Brute force.
    # Phase 2: Medium p (1000 < p < 10^6). Vectorized sieve.
    # Phase 3: Large p (10^6 < p < 10^9). Random sampling + Vectorized sieve.

    best_k = 0
    best_p = 2

    # Helper: Vectorized p-smooth check
    def check_p_smooth_vectorized(p, max_k=200):
        # Returns k such that p^2+1 ... p^2+k are all p-smooth
        # Note: p-smooth means all prime factors <= p.
        
        # We need to divide out all prime factors q <= p.
        # We only have primes up to SIEVE_LIMIT (100000).
        # If p > 100000, we don't have all primes <= p.
        # But if p > 100000, then any factor q of p^2+i that is > 100000 must be checked.
        # Actually, if p is large, p^2 is large.
        # The "remainder" after dividing by small primes (<= 100000) will be 1 or a prime > 100000 or a product of primes > 100000.
        # If the remainder is 1, it's p-smooth (provided all removed factors were <= p).
        # If the remainder > 1:
        #   - If remainder is prime: it must be <= p.
        #   - If remainder is composite: it must have all factors <= p.
        
        # Since we only sieve up to 100000, for p > 100000, we can only divide by primes <= 100000.
        # If p >= 100000, then all divided primes are <= p.
        # The remainder R must be 1. If R > 1, does it have a factor > p?
        # R is composed of primes > 100000.
        # If R is prime, and R > p, fail.
        # If R is composite, say R = a*b. Since a,b > 100000, and R ~ p^2.
        # If p is large (e.g. 10^6), R could be ~ 10^12.
        # If R > 1, we need to verify if all its factors are <= p.
        
        # Let's implement a hybrid:
        # 1. Vectorize division by primes <= min(p, 100000).
        # 2. For the remaining array elements > 1, use slow factorization to check if max factor <= p.
        
        if p > 100000:
             # For large p, the probability that a random number is p-smooth is low?
             # Actually, for p^2+i, the factors are typically around p.
             # If p is large, it's more likely that factors are <= p?
             # Let's stick to the vectorized approach for p < 100000 first.
             pass

        N = np.array([p*p + i for i in range(1, max_k + 1)], dtype=np.int64)
        
        # Primes to divide: all primes q <= p and q <= 100000
        limit_q = min(p, 100000)
        
        # Get primes up to limit_q
        # We can use the precomputed small_primes
        primes_to_use = [q for q in small_primes if q <= limit_q]
        
        # Divide N by each prime
        for q in primes_to_use:
            # While N % q == 0, N = N // q
            # Vectorized:
            mask = N % q == 0
            while np.any(mask):
                N[mask] //= q
                mask = N[mask] % q == 0
        
        # Now N contains the remainders after removing all small prime factors.
        # For each element n_i in N:
        # If n_i == 1, it is p-smooth (assuming we removed all factors <= p).
        # If n_i > 1:
        #   It has no prime factors <= min(p, 100000).
        #   Case 1: p <= 100000. Then we removed all factors <= p.
        #           So any remaining factor is > p. Thus if n_i > 1, it FAILS.
        #   Case 2: p > 100000. We only removed factors <= 100000.
        #           n_i has factors > 100000.
        #           We need to check if all these factors are <= p.
        
        k = 0
        for i in range(max_k):
            if time.time() - start_time > DEADLINE:
                return k
            rem = N[i]
            if rem == 1:
                k += 1
            else:
                if p <= 100000:
                    # All factors <= p were removed. rem > 1 implies a factor > p.
                    break
                else:
                    # p > 100000. rem has factors > 100000.
                    # Check if max prime factor of rem <= p.
                    # Use slow factorization for this single number.
                    factors = _factor(rem)
                    if factors:
                        max_f = max(factors.keys())
                        if max_f > p:
                            break
                    else:
                        # rem should be 1 if factored completely? 
                        # _factor returns {} for 1.
                        # If rem > 1 and _factor returns {}, something is wrong.
                        # But _factor should handle it.
                        pass
                    k += 1
        return k

    # Phase 1: Small primes (up to 1000)
    for p in small_primes:
        if p > 1000:
            break
        if time.time() - start_time > DEADLINE:
            break
        k = check_p_smooth_vectorized(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Medium primes (1000 to 100000)
    # Use vectorized check.
    for p in small_primes:
        if p < 1000:
            continue
        if p > 100000:
            break
        if time.time() - start_time > DEADLINE:
            break
        k = check_p_smooth_vectorized(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 3: Large primes (100000 to 10^9)
    # Random sampling.
    # The vectorized check for p > 100000 is slower because of the fallback factorization.
    # But it's still much faster than checking 200 numbers individually with full factorization.
    
    random.seed(42)
    
    # Try specific ranges
    search_ranges = [
        (100000, 1000000),
        (1000000, 10000000),
        (10000000, 100000000),
        (100000000, 1000000000),
    ]
    
    for lo, hi in search_ranges:
        if time.time() - start_time > DEADLINE:
            break
        num_tried = 0
        max_tries = 1000 # More tries because vectorized check is fast for the "fail early" cases
        while num_tried < max_tries and time.time() - start_time < DEADLINE:
            if hi - lo < 1000:
                p = random.randint(lo, hi)
            else:
                p = random.randint(lo, hi)
            if p % 2 == 0:
                p += 1
            if p < 2:
                continue
            num_tried += 1
            if not _is_prime_mr(p):
                continue
            if p > 10**9:
                continue
            
            k = check_p_smooth_vectorized(p)
            if k > best_k:
                best_k = k
                best_p = p

    # Phase 4: Targeted search near best_p
    if time.time() - start_time < DEADLINE - 10:
        # Search near best_p
        # Use a wider range if best_p is large
        search_width = max(1000, best_p // 100)
        for offset in range(-search_width, search_width + 1):
            if time.time() - start_time > DEADLINE:
                break
            p = best_p + offset
            if p < 2 or p > 10**9:
                continue
            if not _is_prime_mr(p):
                continue
            
            k = check_p_smooth_vectorized(p)
            if k > best_k:
                best_k = k
                best_p = p

    if best_k < 1:
        best_k = 1
        best_p = 2

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
