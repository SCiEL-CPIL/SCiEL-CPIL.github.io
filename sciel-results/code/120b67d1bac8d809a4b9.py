"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 110.0  # safety margin before 150s kill
    
    # Precompute small primes up to 10^5 for trial division (faster startup)
    def sieve_primes(n):
        if n < 2:
            return []
        bs = bytearray(b'\x01') * (n + 1)
        bs[0] = bs[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if bs[i]:
                bs[i*i::i] = bytearray(len(bs[i*i::i]))
        return [i for i in range(2, n + 1) if bs[i]]
    
    small_primes = sieve_primes(10**5)
    
    # Deterministic Miller-Rabin for n < 3.2e18
    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Pollard's rho with Brent's improvement
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            # Use random seed for better distribution
            c = random.randrange(1, n - 1)
            y = random.randrange(0, n - 1)
            m = 128
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g
    
    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p"""
        if n <= 1:
            return True
        if is_prime(n):
            return n <= p
        
        # Trial division by small primes
        for sp in small_primes:
            if sp * sp > n:
                break
            if n % sp == 0:
                if sp > p:
                    return False
                n //= sp
                # Check if remaining n is p-smooth
                if not is_p_smooth(n, p):
                    return False
                return True
        
        if n == 1:
            return True
            
        if is_prime(n):
            return n <= p
            
        # Use Pollard's Rho to find a factor
        d = pollard_brent(n)
        if d == n:
            # Fallback: assume prime (should be caught by is_prime, but just in case)
            return n <= p
            
        # Recursively check factors
        if not is_p_smooth(d, p):
            return False
        if not is_p_smooth(n // d, p):
            return False
        return True
    
    def find_smooth_run(p, max_k=200):
        """Find the longest run of p-smooth numbers starting at p^2+1"""
        base = p * p
        k = 0
        for i in range(1, max_k + 1):
            n = base + i
            if not is_p_smooth(n, p):
                break
            k = i
        return k
    
    # Generate candidate primes and test them
    best_k = 0
    best_p = 2
    
    # 1. Test small primes first (fast factorization)
    for p in small_primes:
        if time.time() > deadline - 5:
            break
        k = find_smooth_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # 2. Search larger primes using segmented sieve
    def segmented_sieve(low, high):
        """Generate primes in [low, high]"""
        if high < 2:
            return []
        limit = int(math.isqrt(high)) + 1
        # Reuse small primes if possible, else sieve
        if limit <= len(small_primes) and small_primes[-1] >= limit:
             small = [p for p in small_primes if p <= limit]
        else:
             small = sieve_primes(limit)
             
        sieve = bytearray(b'\x01') * (high - low + 1)
        for p in small:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                sieve[j - low] = 0
        primes = []
        for i in range(low, high + 1):
            if sieve[i - low] and i >= 2:
                primes.append(i)
        return primes
    
    # Search ranges
    # Start from 10^5 to 10^6, then 10^6 to 10^7, etc.
    # Adjust step size based on time
    current_low = 10**5
    current_high = 10**6
    
    while current_high <= 10**9 and time.time() < deadline - 10:
        candidates = segmented_sieve(current_low, current_high)
        
        for p in candidates:
            if time.time() > deadline - 5:
                break
            
            # Quick pre-check: test first 2 numbers
            base = p * p
            if not is_p_smooth(base + 1, p):
                continue
            if not is_p_smooth(base + 2, p):
                continue
            
            # If pre-check passes, do full check
            k = find_smooth_run(p)
            if k > best_k:
                best_k = k
                best_p = p
        
        # Increase range
        current_low = current_high
        current_high = current_high * 10 # Jump by order of magnitude
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
