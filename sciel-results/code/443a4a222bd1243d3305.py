"""Rewritten program for long runs after p^2 with p-smooth factors."""
import math
import time
import random

# EVOLVE-BLOCK-START

def _is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.2e18."""
    if n < 2:
        return False
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        return True
    if n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
        return False
    # Small prime trial division
    for p in (7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def _pollard_brent(n):
    """Pollard's Brent variant for finding a factor of n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), 128
    g, r, q = 1, 1, 1
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
    return g

def _factor(n, factors=None):
    """Fully factor n using trial division + Pollard-Brent + Miller-Rabin."""
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if _is_prime_mr(n):
        factors.append(n)
        return factors
    d = _pollard_brent(n)
    _factor(d, factors)
    _factor(n // d, factors)
    return factors

def _largest_prime_factor(n):
    """Return the largest prime factor of n."""
    if n == 1:
        return 1
    # Quick trial division by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if n % p == 0:
            n //= p
            while n % p == 0:
                n //= p
            if n == 1:
                return p
            # Need to check if remaining n has larger factor
            lf = _largest_prime_factor(n)
            return max(p, lf)
    # n is not divisible by any small prime
    if _is_prime_mr(n):
        return n
    # Use full factorization
    factors = _factor(n)
    return max(factors)

def _largest_factor_fast(n, limit):
    """Check if all prime factors of n are <= limit. Return largest factor if > limit, else -1."""
    if n == 1:
        return -1
    # Trial divide by small primes up to min(sqrt(n), limit)
    # But we only need to know if there's a factor > limit
    # Strategy: divide out all factors <= limit; if remainder > 1, it has a factor > limit
    temp = n
    # Trial division by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    for p in small_primes:
        if p > limit:
            break
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            if temp == 1:
                return -1
    if temp == 1:
        return -1
    # temp is not divisible by any prime <= min(limit, 199)
    # If temp <= limit, it's prime (or 1) and thus all factors <= limit
    if temp <= limit:
        return -1
    # temp > limit and temp has no factor <= 199
    # Check if temp is prime
    if _is_prime_mr(temp):
        return temp  # temp is prime > limit
    # temp is composite > limit
    # We need to find if it has a factor > limit
    # Since temp > limit and temp is composite, if all factors were <= limit, 
    # then temp would need to be a product of primes all <= limit
    # But we've already divided out all primes <= 199. If limit >= 199, 
    # we need to check more.
    # Simpler: just get the largest prime factor
    factors = _factor(temp)
    lf = max(factors)
    return lf if lf > limit else -1

def _sieve_primes(limit):
    """Sieve of Eratosthenes to generate primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _check_run(p, k, deadline):
    """Check if p^2+1 through p^2+k all have prime factors <= p.
    Return the actual run length found."""
    p2 = p * p
    for i in range(1, k + 1):
        if time.time() > deadline:
            return i - 1
        n = p2 + i
        # Check if all prime factors of n are <= p
        # Divide out small primes first
        temp = n
        # Trial division by primes up to min(sqrt(temp), p)
        # But we can be smarter: if after dividing out all primes <= p, temp == 1, we're good
        # However, p can be large, so trial division up to p is too slow for large p
        # Instead: divide out small primes, then check remainder
        for d in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if d > p:
                break
            if temp % d == 0:
                while temp % d == 0:
                    temp //= d
                if temp == 1:
                    break
        if temp == 1:
            continue
        # temp is not 1 and not divisible by primes up to 97 (or up to p if p < 97)
        if temp <= p:
            # temp is either prime or composite, but all its factors are <= temp <= p
            continue
        # temp > p
        # If temp is prime, it's a factor > p, so fail
        if _is_prime_mr(temp):
            return i - 1
        # temp is composite > p
        # Check if all prime factors of temp are <= p
        # We need to verify this. Use partial factorization.
        # Try to find a factor of temp that is > p
        # If temp has a factor d where d > p and temp//d > p, or if temp itself has a prime factor > p
        # Actually, we just need to check: does temp have any prime factor > p?
        # Since temp > p, if temp is a product of primes all <= p, that's fine.
        # Let's try to factor temp partially
        # First, check if temp is a perfect square of a prime, etc.
        # Simpler: use _largest_prime_factor but with early termination
        lf = _largest_prime_factor(temp)
        if lf > p:
            return i - 1
        # All factors of temp are <= p, so n is p-smooth
    return k

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Small primes exhaustive (p up to 10000)
    primes_small = _sieve_primes(10000)
    for p in primes_small:
        if time.time() > deadline:
            break
        if p < 3:
            continue
        # Check run length
        p2 = p * p
        k = 0
        max_k_check = 200
        while k < max_k_check:
            if time.time() > deadline:
                break
            n = p2 + k + 1
            # Check if n is p-smooth
            temp = n
            # Trial division by small primes
            for d in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                if d > p:
                    break
                if temp % d == 0:
                    while temp % d == 0:
                        temp //= d
                    if temp == 1:
                        break
            if temp == 1:
                k += 1
                continue
            if temp <= p:
                k += 1
                continue
            # temp > p
            if _is_prime_mr(temp):
                break
            # temp is composite > p, check largest factor
            lf = _largest_prime_factor(temp)
            if lf > p:
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break
    
    # Phase 2: Medium primes systematic (p from 10000 to 1000000)
    # Use a larger sieve or generate primes on the fly
    # Sieve up to 1000000
    if time.time() < deadline - 20:
        primes_med = _sieve_primes(1000000)
        for p in primes_med:
            if p <= 10000:
                continue
            if time.time() > deadline:
                break
            if p < 3:
                continue
            p2 = p * p
            k = 0
            max_k_check = 200
            while k < max_k_check:
                if time.time() > deadline:
                    break
                n = p2 + k + 1
                temp = n
                for d in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                    if d > p:
                        break
                    if temp % d == 0:
                        while temp % d == 0:
                            temp //= d
                        if temp == 1:
                            break
                if temp == 1:
                    k += 1
                    continue
                if temp <= p:
                    k += 1
                    continue
                if _is_prime_mr(temp):
                    break
                lf = _largest_prime_factor(temp)
                if lf > p:
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
                if best_k >= 200:
                    break
    
    # Phase 3: Large primes random sampling (p up to 10^9)
    if time.time() < deadline - 5:
        # Generate random odd numbers and check if prime
        # Focus on ranges that might yield longer runs
        # Heuristic: larger p tends to have longer runs on average
        random.seed(42)
        attempts = 0
        while time.time() < deadline - 2 and attempts < 5000:
            attempts += 1
            # Sample from various ranges
            if random.random() < 0.3:
                p = random.randint(10**8, 10**9)
            elif random.random() < 0.6:
                p = random.randint(10**6, 10**8)
            else:
                p = random.randint(10**5, 10**6)
            
            # Make odd
            if p % 2 == 0:
                p += 1
            
            # Check if prime
            if not _is_prime_mr(p):
                continue
            
            p2 = p * p
            k = 0
            max_k_check = 200
            while k < max_k_check:
                if time.time() > deadline:
                    break
                n = p2 + k + 1
                temp = n
                for d in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                    if d > p:
                        break
                    if temp % d == 0:
                        while temp % d == 0:
                            temp //= d
                        if temp == 1:
                            break
                if temp == 1:
                    k += 1
                    continue
                if temp <= p:
                    k += 1
                    continue
                if _is_prime_mr(temp):
                    break
                lf = _largest_prime_factor(temp)
                if lf > p:
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
                if best_k >= 200:
                    break
    
    # Ensure valid result
    if best_k < 1:
        best_k = 1
        best_p = 2
    
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
