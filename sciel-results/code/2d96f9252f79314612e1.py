import math
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    # 1. Precompute small primes for trial division
    SIEVE_LIMIT = 10**6
    def simple_sieve(n):
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                is_prime[i * i: n + 1: i] = False
        return np.array(np.where(is_prime)[0])

    small_primes = simple_sieve(SIEVE_LIMIT)

    # 2. Miller-Rabin for primality test of larger numbers
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases sufficient for n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 3. Pollard's Rho with Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
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
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    # 4. Check if n is p-smooth
    # p_limit is the prime p we are testing against
    def is_p_smooth(n, p_limit):
        if n <= 1:
            return True
        
        # If n is small enough, we can just check primes up to min(p_limit, sqrt(n))
        # But we use the precomputed small_primes up to 10^6.
        
        # Determine the slice of small_primes to use
        # We only need primes <= p_limit
        idx = np.searchsorted(small_primes, p_limit)
        primes_to_try = small_primes[:idx]
        
        if len(primes_to_try) == 0:
            # p_limit is very small (e.g., 2)
            # Check if n is prime and <= p_limit
            return is_prime(n) and n <= p_limit

        # Vectorized check for divisibility
        # Note: n might be large (up to 10^18)
        # n % primes_to_try works with NumPy if n is a scalar? 
        # Actually, n % array works.
        
        rem = n % primes_to_try
        mask = rem == 0
        
        if not np.any(mask):
            # No small prime factors found
            if p_limit < SIEVE_LIMIT:
                # All primes <= p_limit were tried. None divided n.
                # So n has no prime factor <= p_limit.
                # Thus n is not p-smooth (unless n=1, handled above).
                return False
            else:
                # p_limit >= SIEVE_LIMIT.
                # n has no prime factor <= 10^6.
                # Check if n is prime.
                if is_prime(n):
                    return n <= p_limit
                else:
                    # n is composite.
                    # Find a factor using Pollard Rho
                    d = pollard_brent(n)
                    return is_p_smooth(d, p_limit) and is_p_smooth(n // d, p_limit)
        
        # Divide out the small factors
        # We iterate over the factors found
        # To avoid iterating over all primes, we only iterate over the mask
        factors = primes_to_try[mask]
        for p in factors:
            while n % p == 0:
                n //= p
        
        if n == 1:
            return True
        
        # Recurse on the remaining part
        return is_p_smooth(n, p_limit)

    best_k = 0
    best_p = 2

    # Search for primes p
    # We use a segmented sieve to generate primes p
    # We'll search up to a certain limit, or until time runs out.
    # Let's search in segments of size 10^6.
    
    segment_size = 10**6
    low = 2
    high = low + segment_size
    
    # Sieve small primes for the segmented sieve (primes up to sqrt(high))
    # We already have small_primes up to 10^6, which is enough for sqrt(10^12)
    # But high can go up to 10^9. sqrt(10^9) ~ 31623.
    # small_primes covers this.
    
    while time.time() < deadline and low < 10**9:
        if time.time() > deadline - 5:
            break
            
        # Segmented sieve for [low, high)
        # Initialize array
        size = high - low
        is_prime_seg = np.ones(size, dtype=bool)
        
        # Sieve with small primes
        for p in small_primes:
            if p * p >= high:
                break
            # Find first multiple of p in [low, high)
            start = max(p * p, ((low + p - 1) // p) * p)
            if start < low:
                start += p
            # Mark multiples
            is_prime_seg[start - low: high - low: p] = False
        
        # Extract primes
        primes_in_seg = np.where(is_prime_seg)[0] + low
        
        # Filter out non-primes (if low < 2)
        if low < 2:
            primes_in_seg = primes_in_seg[primes_in_seg >= 2]
        
        # Check each prime
        for p in primes_in_seg:
            if time.time() > deadline - 2:
                break
                
            # Check k for this p
            k = 0
            p_sq = p * p
            while k < 200:
                if time.time() > deadline - 2:
                    break
                v = p_sq + k + 1
                if is_p_smooth(v, p):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
                
        low = high
        high += segment_size
        
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
