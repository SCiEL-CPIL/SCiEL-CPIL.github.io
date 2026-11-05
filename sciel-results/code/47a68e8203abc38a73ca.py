import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 112.0

    # Precompute primes up to a reasonable limit for trial division
    SIEVE_LIMIT = 5_000_000
    is_prime_sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, SIEVE_LIMIT + 1, i):
                is_prime_sieve[j] = 0
    primes_list = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime_sieve[i]]
    
    # Miller-Rabin for 64-bit
    def is_prime_mr(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in bases:
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

    # Pollard's Rho with Brent's improvement
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
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
                    if g > 1:
                        break
            if g != n:
                return g

    def factorize(n):
        """Fully factorize n, returning a list of prime factors."""
        if n <= 1:
            return []
        if is_prime_mr(n):
            return [n]
        # Try small primes first
        for p in primes_list[:100]:
            if n % p == 0:
                result = [p]
                while n % p == 0:
                    n //= p
                result.extend(factorize(n))
                return result
        # Use Pollard's Rho
        d = pollard_brent(n)
        result = []
        if d == n:
            # Fallback: try again or use trial division for small factors
            for p in primes_list:
                if p * p > n:
                    break
                if n % p == 0:
                    result.append(p)
                    while n % p == 0:
                        n //= p
            if n > 1:
                result.append(n)
            return result
        result.extend(factorize(d))
        result.extend(factorize(n // d))
        return result

    def is_p_smooth_fast(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= p:
            return True
        if is_prime_mr(n):
            return False  # n is prime and > p
        
        factors = factorize(n)
        for f in factors:
            if f > p:
                return False
        return True

    def quick_p_smooth_check(n, p, small_primes):
        """Quick check: divide out small primes, then check remainder."""
        temp = n
        for q in small_primes:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
        if temp == 1:
            return True
        if temp <= p:
            return True
        if is_prime_mr(temp):
            return False  # temp is prime > p
        # Need full factorization
        return None  # Unknown

    best_k = 1
    best_p = 2
    
    # Strategy: 
    # 1. First, do a fast pre-sieve over all primes to find candidates where p^2+1 is p-smooth
    # 2. Then do full verification on those candidates
    # 3. Use random sampling for larger primes
    
    # Phase 1: Fast pre-sieve - check only p^2+1 with small prime division
    # If p^2+1 has a prime factor > p, skip this p entirely
    candidates = []
    
    # Use a subset of small primes for quick check (up to ~10000)
    quick_primes = [q for q in primes_list if q <= 10000]
    
    for idx, p in enumerate(primes_list):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        p2 = p * p
        n = p2 + 1
        
        # Quick check: does n have all prime factors <= p?
        result = quick_p_smooth_check(n, p, quick_primes)
        if result is True:
            candidates.append(p)
        # If result is False or None, skip for now
    
    # Phase 2: Full verification on candidates, sorted by potential
    # Try candidates in order, but also interleave with random large primes
    random.seed(42)
    
    # Sort candidates: try smaller ones first (more likely to have longer runs due to density)
    # Actually, let's try a mix: first pass small candidates, then large
    candidates.sort()
    
    # Also generate some random large primes to check
    large_prime_candidates = []
    if time.time() - start_time < DEADLINE - 40:
        # Sample some large primes
        for _ in range(min(200, len(primes_list) // 50)):
            idx = random.randint(len(primes_list) // 2, len(primes_list) - 1)
            p = primes_list[idx]
            if p not in candidates:
                large_prime_candidates.append(p)
    
    # Interleave: check candidates and large primes
    all_to_check = list(candidates) + large_prime_candidates
    
    for p in all_to_check:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        p2 = p * p
        k = 0
        for j in range(1, 201):
            n = p2 + j
            if is_p_smooth_fast(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we still have time, do a more thorough search
    if time.time() - start_time < DEADLINE - 10 and len(candidates) > 0:
        # Re-verify top candidates more carefully, and try nearby primes
        for p in candidates[:50]:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Check p and nearby primes
            p_idx = primes_list.index(p)
            for dp in [-2, -1, 0, 1, 2]:
                if p_idx + dp < 0 or p_idx + dp >= len(primes_list):
                    continue
                p_try = primes_list[p_idx + dp]
                if p_try == p:
                    continue
                
                p2 = p_try * p_try
                k = 0
                for j in range(1, 201):
                    n = p2 + j
                    if is_p_smooth_fast(n, p_try):
                        k += 1
                    else:
                        break
                
                if k > best_k:
                    best_k = k
                    best_p = p_try
    
    return {'p': best_p, 'k': best_k}
