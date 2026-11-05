"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Check for small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Miller-Rabin
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
    
    # Pollard's Rho with Brent's cycle detection
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
    
    # Factorize n, returning list of prime factors (with multiplicity)
    def factorize(n):
        factors = []
        # Try small primes first
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        # Use Pollard's Rho for remaining composite
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                factors.append(m)
            else:
                d = pollard_brent(m)
                if d == m:
                    # Retry
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return factors
    
    # Check if all prime factors of v are <= limit
    def all_factors_leq(v, limit):
        if v <= 1:
            return True
        # Quick check: if v is prime and > limit, fail
        if is_prime(v):
            return v <= limit
        # Factorize
        factors = factorize(v)
        return all(f <= limit for f in factors)
    
    best_k = 0
    best_p = 2
    
    # Search primes p
    # We need to be efficient. Let's search p in a reasonable range.
    # The seed searched up to 2000. Let's extend that but with a time guard.
    
    # Generate primes up to some limit, or just iterate and check primality
    # Since p can be up to 10^9, we can't iterate all. But large p means p^2 is huge.
    # For small p, p^2 + k might have small factors.
    # Let's search p from 3 upwards, checking primality with Miller-Rabin.
    
    p = 3
    while time.time() < deadline and p < 10**9:
        if is_prime(p):
            k = 0
            # Check p^2 + 1, p^2 + 2, ...
            # We want to maximize k
            # Note: p^2 + 1 is odd for odd p, so not divisible by 2
            # For each candidate, check if all prime factors <= p
            # We can stop early if we find a factor > p
            while time.time() < deadline:
                v = p * p + k + 1
                if all_factors_leq(v, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
        p += 2  # Only check odd numbers (2 is handled separately, but k for p=2 is small)
    
    # Also check p = 2
    if time.time() < deadline:
        k = 0
        while time.time() < deadline:
            v = 4 + k + 1
            if all_factors_leq(v, 2):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = 2
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
