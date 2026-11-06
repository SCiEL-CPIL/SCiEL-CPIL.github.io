"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Deterministic Miller-Rabin for n < 2^64
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
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
    
    # Pollard's Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, 100)
            g = r = q = 1
            x = ys = 0
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
    
    def factorize(n, factors=None):
        if factors is None:
            factors = []
        if n <= 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        d = pollard_brent(n)
        factorize(d, factors)
        factorize(n // d, factors)
        return factors
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        if is_prime(n):
            return n
        # Try small factors first
        for d in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            while n % d == 0:
                n //= d
        if n <= 1:
            return 1
        if is_prime(n):
            return n
        factors = factorize(n)
        return max(factors)
    
    # Sieve for small primes
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit + 1, i):
                    is_p[j] = False
        return [i for i, v in enumerate(is_p) if v]
    
    best_k = 1
    best_p = 2
    
    # First, find some small primes to get a baseline
    small_primes = sieve_primes(10000)
    
    for p in small_primes:
        if time.time() > deadline:
            break
        # Check if p^2 + 1 is prime (if so, k=0)
        if is_prime(p * p + 1):
            continue
        k = 0
        while k < 200:
            val = p * p + k + 1
            lpf = largest_prime_factor(val)
            if lpf > p:
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
    
    # Now search larger primes using random sampling
    # We'll search in ranges, using Miller-Rabin to filter
    search_range = 10**7
    
    # Search in chunks
    for start in range(10000, search_range, 10000):
        if time.time() > deadline:
            break
        # Generate candidate primes in this range
        end = min(start + 10000, search_range)
        # Use a simple approach: check odd numbers
        for candidate in range(start, end, 2):
            if time.time() > deadline:
                break
            if not is_prime(candidate):
                continue
            p = candidate
            # Quick check: if p^2+1 is prime, skip
            if is_prime(p * p + 1):
                continue
            k = 0
            while k < 200:
                val = p * p + k + 1
                lpf = largest_prime_factor(val)
                if lpf > p:
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
    
    # If we have time, do random search in larger ranges
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        for _ in range(min(100, int(remaining_time * 10))):
            if time.time() > deadline:
                break
            # Random prime candidate
            p = random.randint(10**6, 10**9)
            # Round to odd
            if p % 2 == 0:
                p += 1
            # Find next prime
            while not is_prime(p):
                p += 2
            if p > 10**9:
                break
            # Quick check
            if is_prime(p * p + 1):
                continue
            k = 0
            while k < 200:
                val = p * p + k + 1
                lpf = largest_prime_factor(val)
                if lpf > p:
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
