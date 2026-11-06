import time
import math
import random

def is_prime(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
    # Bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # For n < 10^9, bases 2, 3, 5, 7, 11 are enough.
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13]:
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n)
        c = random.randrange(1, n)
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

def factorize(n):
    factors = []
    def _factorize(n):
        if n == 1:
            return
        if is_prime(n):
            factors.append(n)
            return
        d = pollard_brent(n)
        _factorize(d)
        _factorize(n // d)
    _factorize(n)
    return factors

def sieve(limit):
    is_prime_flag = [True] * (limit + 1)
    is_prime_flag[0] = is_prime_flag[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_flag[i]:
            for j in range(i*i, limit + 1, i):
                is_prime_flag[j] = False
    return [i for i, prime in enumerate(is_prime_flag) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    # Sieve up to 10^7
    # 10^7 is ~10MB boolean array
    limit = 10**7
    primes = sieve(limit)
    
    best_k = 0
    best_p = 2
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Check run length for this p
        k = 0
        # We need to check p^2 + 1, ..., p^2 + k
        # If any has a prime factor > p, stop.
        valid = True
        while k < 200:
            n = p * p + k + 1
            # Factorize n
            # Optimization: if n is prime, factor is n. Check n <= p?
            # If n is prime and n > p, fail.
            # If n is composite, check factors.
            
            # Quick check: if n is prime
            if is_prime(n):
                if n > p:
                    valid = False
                    break
                else:
                    k += 1
                    continue
            
            # Otherwise, factorize
            factors = factorize(n)
            # Check if any factor > p
            fail = False
            for f in factors:
                if f > p:
                    fail = True
                    break
            if fail:
                valid = False
                break
            else:
                k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
