import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n == 2 or n == 3
        # Deterministic Miller-Rabin for 64-bit
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
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
                        q = (q * abs(x - y)) % n
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

    # Pre-compute small primes for trial division
    limit = 100000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Filter small primes up to a reasonable bound for quick division
    # We will use primes up to 1000 for the "screening" step
    quick_primes = [p for p in small_primes if p <= 1000]

    def has_large_factor(n, p):
        """Returns True if n has any prime factor > p"""
        if n <= 1:
            return False
        if p < 2:
            return True
        
        # Quick trial division by very small primes
        for sp in quick_primes:
            if sp > p:
                break
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return False
        
        if n == 1:
            return False
        
        # If n is prime, check if it exceeds p
        if is_prime(n):
            return n > p
        
        # n is composite. We need to find if it has a factor > p.
        # Strategy: Factor n completely? Or just find one factor > p?
        # If we find a factor d such that d > p, return True.
        # If we find a factor d such that d <= p, we must check n/d.
        # This is effectively finding the largest prime factor.
        
        # Optimization: If n > p*p, it's not necessarily true that it has a factor > p.
        # Example: n = (p-1)*(p-1). Factors are p-1 <= p.
        # But if n has a prime factor q > p, then q must be a factor of n.
        
        # Let's recursively factor.
        # Since we've divided out small primes, n is likely a product of larger primes.
        
        factors = []
        temp_n = n
        while temp_n > 1:
            if is_prime(temp_n):
                factors.append(temp_n)
                break
            d = pollard_brent(temp_n)
            if d == temp_n:
                # Should not happen, but just in case
                continue
            factors.append(d)
            temp_n //= d
            while temp_n % d == 0:
                temp_n //= d
                factors.append(d)
        
        # Check if any factor > p
        for f in factors:
            if f > p:
                return True
        return False

    def check_run(p, max_k=200):
        """Check the longest run of p-smooth numbers starting from p^2+1"""
        p2 = p * p
        k = 0
        while k < max_k:
            val = p2 + k + 1
            if has_large_factor(val, p):
                break
            k += 1
            # Time check
            if time.time() - start_time > DEADLINE:
                break
        return k

    best_k = 0
    best_p = 2
    
    # Strategy:
    # 1. Scan a range of smaller primes to establish a baseline and catch any unexpectedly long runs.
    # 2. Use random search for larger primes, as long runs are rare and likely require larger p 
    #    to have enough "room" for factors to stay small relative to p.
    #    Actually, the probability of a number n being p-smooth is roughly log(p)/log(n).
    #    For n ~ p^2, this is ~ 1/2. So for k=1, ~50% chance. For k=2, ~25%.
    #    So we expect to find k=2 fairly easily, k=3 less so.
    #    To find k=10, we need p to be large enough that p-smoothness is more likely?
    #    Wait, as p increases, the density of p-smooth numbers in [p^2, p^2+k] is roughly (log p / log p^2)^k = (1/2)^k.
    #    So the probability of a run of length k is roughly constant with respect to p (for large p).
    #    This suggests random search over a wide range of p is effective.
    
    # Let's do a quick scan of primes up to 10^6 first.
    # Then switch to random primes up to 10^9.
    
    # Phase 1: Small primes scan
    # We'll iterate through odd numbers and check primality.
    # To speed this up, we can use a segmented sieve or just Miller-Rabin.
    # Miller-Rabin is fast for 64-bit.
    
    current_p = 1000
    while current_p < 1000000 and time.time() - start_time < DEADLINE:
        if current_p % 2 == 0:
            current_p += 1
        else:
            current_p += 2
            
        if is_prime(current_p):
            p = current_p
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
                
    # Phase 2: Random large primes
    # Pick random 30-bit numbers, make them odd, check prime.
    # We will sample many candidates.
    
    # To avoid duplicates and ensure coverage, we could use a set, but for random search,
    # collisions are unlikely in this range if we just pick random.
    
    # Let's estimate how many we can check.
    # check_run involves has_large_factor which uses Pollard-Brent for composites.
    # If k=0 (p^2+1 has large factor), it's fast (just is_prime or one pollard).
    # If k is small, it's fast.
    # So we can check many candidates.
    
    while time.time() - start_time < DEADLINE:
        # Generate random candidate in [10^6, 10^9]
        # 10^9 is 2^30 approx.
        # Let's pick random 30-bit odd numbers.
        p = random.randint(10**6, 10**9)
        p |= 1  # Make odd
        
        if is_prime(p):
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
                
    return {'p': best_p, 'k': best_k}
