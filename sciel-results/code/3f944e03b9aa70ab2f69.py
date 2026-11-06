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
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
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
    
    # Check if all prime factors of v are <= limit
    def all_factors_leq(v, limit):
        if v <= 1:
            return True
        if is_prime(v):
            return v <= limit
        n = v
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            while n % p == 0:
                if p > limit:
                    return False
                n //= p
        if n == 1:
            return True
        if is_prime(n):
            return n <= limit
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
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return True

    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                start = i * i
                is_p[start:limit+1:i] = bytearray(len(range(start, limit+1, i)))
        return [i for i in range(2, limit+1) if is_p[i]]
    
    # Pre-sieve primes up to 10^6 for fast division
    SIEVE_LIMIT = 10**6
    small_primes = sieve(SIEVE_LIMIT)
    
    # Batch check: for a given p, find the longest run k such that
    # all of p^2+1, ..., p^2+k are p-smooth
    def batch_run_length(p, max_k=200):
        if p == 2:
            # Special case: only 2-smooth numbers (powers of 2)
            # p^2 = 4, check 5,6,7,...
            k = 0
            v = 4 + 1
            while k < max_k:
                # v must be a power of 2
                if v & (v - 1) == 0 and v > 0:
                    k += 1
                    v += 1
                else:
                    break
            return k
        
        # For p > 2: divide out all small prime factors <= p
        # We'll process the range [p^2+1, p^2+max_k]
        start = p * p + 1
        # Create array of values
        vals = [start + i for i in range(max_k)]
        
        # Sieve out small primes up to min(p, SIEVE_LIMIT)
        for q in small_primes:
            if q > p:
                break
            # Find first multiple of q in [start, start+max_k)
            first = ((start + q - 1) // q) * q
            if first > start + max_k - 1:
                continue
            for idx in range(int(first - start), max_k, q):
                while vals[idx] % q == 0:
                    vals[idx] //= q
        
        # After sieving, check remaining values
        # A value is p-smooth iff it equals 1 after removing all factors <= p
        # If vals[i] > 1, it has a prime factor > SIEVE_LIMIT (and possibly > p)
        # We need to check if that factor is <= p
        k = 0
        for i in range(max_k):
            v = vals[i]
            if v == 1:
                k = i + 1
                continue
            # v has a prime factor > SIEVE_LIMIT
            # Check if v (or its prime factors) are all <= p
            if all_factors_leq(v, p):
                k = i + 1
            else:
                break
        return k
    
    best_k = 0
    best_p = 2
    
    # Check small primes first (faster to evaluate)
    # Start with p=2
    k2 = batch_run_length(2)
    if k2 > best_k:
        best_k = k2
        best_p = 2
    
    # Search primes up to 10^6
    for p in small_primes:
        if p == 2:
            continue
        if time.time() >= deadline:
            break
        k = batch_run_length(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we have time, search larger primes
    if time.time() < deadline - 5:
        limit2 = 10**7
        primes2 = sieve(limit2)
        start_idx = len(small_primes)
        for p in primes2[start_idx:]:
            if time.time() >= deadline:
                break
            k = batch_run_length(p)
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
