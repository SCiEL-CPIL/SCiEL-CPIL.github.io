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
    
    # Check if all prime factors of v are <= limit, with early termination
    def all_factors_leq(v, limit):
        if v <= 1:
            return True
        n = v
        # Try small primes first
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
        for p in small_primes:
            if p > limit:
                break
            while n % p == 0:
                n //= p
        if n == 1:
            return True
        if is_prime(n):
            return n <= limit
        # Use Pollard's Rho for remaining composite
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                if m > limit:
                    return False
            else:
                d = pollard_brent(m)
                if d == m:
                    # Retry
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return True
    
    best_k = 0
    best_p = 2
    
    # Sieve primes up to a limit to iterate efficiently
    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                step = i
                start = i * i
                is_p[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        return [i for i in range(2, limit+1) if is_p[i]]
    
    # Check p = 2
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

    # Search primes p > 2
    limit = 200000
    primes = sieve(limit)
    
    for p in primes:
        if time.time() >= deadline:
            break
        if p == 2:
            continue
        # Quick filter: check if p^2+1 is p-smooth first
        if not all_factors_leq(p * p + 1, p):
            continue
        k = 1
        while time.time() < deadline and k < 200:
            v = p * p + k + 1
            if all_factors_leq(v, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            
    # If we still have time, search more primes
    if time.time() < deadline - 5:
        limit2 = 500000
        primes2 = sieve(limit2)
        start_idx = len(primes)
        for p in primes2[start_idx:]:
            if time.time() >= deadline:
                break
            # Quick filter: check if p^2+1 is p-smooth first
            if not all_factors_leq(p * p + 1, p):
                continue
            k = 1
            while time.time() < deadline and k < 200:
                v = p * p + k + 1
                if all_factors_leq(v, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
