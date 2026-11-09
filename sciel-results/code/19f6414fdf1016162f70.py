"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses optimized Pollard's rho + Miller-Rabin for fast factorization, and searches
primes efficiently to maximize k within the time budget."""

import time
import random
import math

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
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
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------- Pollard's Rho with Brent's improvement ----------
def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        x = random.randrange(2, n)
        y = x
        c = random.randrange(1, n)
        d = 1
        # Brent's cycle detection
        q = 1
        r = 1
        m = 128
        ys = y
        while d == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and d == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                d = math.gcd(q, n)
                k += m
            r *= 2
        if d != n:
            return d
        # If d == n, restart with different parameters
        d = 1
        while d == 1:
            ys = (ys * ys + c) % n
            d = math.gcd(abs(x - ys), n)
        if d != n:
            return d

# ---------- Factorization ----------
def factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_rho(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def all_factors_at_most_p(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    # First try small primes
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        while n % q == 0:
            if q > p:
                return False
            n //= q
    if n == 1:
        return True
    # Quick check: if n > p and is_prime(n), then n is a prime factor > p
    if n > p and is_prime(n):
        return False
    # Use Pollard-Rho for the rest
    factors = factor(n)
    for f in factors:
        if f > p:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Precompute small primes for trial division
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def check_prime(n):
        if n < 2:
            return False
        if n in small_primes:
            return True
        if n % 2 == 0:
            return False
        for q in small_primes[1:]:
            if q * q > n:
                break
            if n % q == 0:
                return False
        return is_prime(n)
    
    # Strategy: 
    # 1. First, quickly check small primes (up to ~1000) to get a baseline
    # 2. Then, search larger primes more selectively
    # Key insight: for a prime p, p^2 + i for small i often has factors <= p
    # because p^2 + i = (p + something)(p - something) + something
    # Actually, let's just do a systematic search
    
    # Try p=2 first
    k = 0
    while k < 200 and all_factors_at_most_p(4 + k + 1, 2):
        k += 1
    if k > best_k:
        best_k = k
        best_p = 2
    
    # Systematic search over primes
    # We'll search in increasing order, but with adaptive step sizes
    # For small p, we can check many quickly
    # For large p, factorization is slower but runs might be longer
    
    # Phase 1: Check primes up to 10000 quickly
    for p in range(3, 10000, 2):
        if time.time() > deadline:
            break
        if not check_prime(p):
            continue
        
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() > deadline:
                break
            if not all_factors_at_most_p(p_sq + k + 1, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Check larger primes up to 100000
    # Use a coarser step but still check primality
    for p in range(10001, 100000, 2):
        if time.time() > deadline:
            break
        if not check_prime(p):
            continue
        
        k = 0
        p_sq = p * p
        # For larger p, we might not be able to check all 200 values
        # But let's try
        while k < 200:
            if time.time() > deadline:
                break
            if not all_factors_at_most_p(p_sq + k + 1, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 3: If we have time, try some larger primes
    # For very large p, the numbers p^2 + i are large, but more likely to have small factors
    # Let's try primes in the range 100000 to 10^6 with a coarser approach
    for p in range(100001, 1000000, 2):
        if time.time() > deadline:
            break
        if not check_prime(p):
            continue
        
        k = 0
        p_sq = p * p
        # Limit the check to save time for larger p
        max_check = min(200, 50)  # Check at most 50 values for large p
        while k < max_check:
            if time.time() > deadline:
                break
            if not all_factors_at_most_p(p_sq + k + 1, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 4: If we still have time, try even larger primes
    # But factorization becomes very slow for p > 10^6
    # Let's try a few random large primes
    if time.time() < deadline - 10:
        # Try some specific larger primes that might have good properties
        # These are primes where p^2 + i might have many small factors
        large_primes_to_try = []
        # Generate a few candidates
        for base in [10**6, 2*10**6, 5*10**6, 10**7, 5*10**7, 10**8]:
            for offset in range(-100, 101, 2):
                candidate = base + offset
                if candidate > 0 and candidate <= 10**9 and candidate % 2 != 0:
                    if check_prime(candidate):
                        large_primes_to_try.append(candidate)
        
        # Try these primes
        for p in large_primes_to_try[:20]:  # Limit to 20 large primes
            if time.time() > deadline - 5:
                break
            k = 0
            p_sq = p * p
            # For very large p, limit the check
            max_check = min(200, 30)
            while k < max_check:
                if time.time() > deadline:
                    break
                if not all_factors_at_most_p(p_sq + k + 1, p):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
