import time
import random
from math import gcd

def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
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
        y = random.randint(1, n - 1)
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
                g = gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
        if g != n:
            return g

def is_p_smooth(n, p):
    if n == 1:
        return True
    if is_prime(n):
        return n <= p
    # Try small primes first
    for q in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if n % q == 0:
            if q > p:
                return False
            return is_p_smooth(n // q, p)
    d = pollard_brent(n)
    if d == n:
        # Should not happen if n is composite, but handle just in case
        return is_prime(n) and n <= p
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_p = 2
    best_k = 1 # p=2, p^2+1=5 (prime > 2, fail). p=3, 9+1=10=2*5 (5>3 fail). 
               # p=5, 25+1=26=2*13 (13>5 fail).
               # p=7, 49+1=50=2*5*5 (max 5 <= 7). 49+2=51=3*17 (17>7 fail). k=1.
               # Let's just find the max.
    
    # We need to find p such that p^2+1 ... p^2+k are p-smooth.
    
    # Iterate through primes
    # Simple prime generation
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, prime in enumerate(sieve) if prime]

    # Start with small primes, expand if time allows
    # But p can be up to 1e9. 
    # However, factorization of p^2+j for large p is expensive.
    # Small p is faster to check.
    # Let's scan p up to 10^5 first.
    
    primes = get_primes(100000)
    
    for p in primes:
        if time.time() > deadline:
            break
            
        k = 0
        # Check j from 1 upwards
        while True:
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
                if k > best_k:
                    best_k = k
                    best_p = p
            else:
                break
            if time.time() > deadline:
                break
                
    return {'p': best_p, 'k': best_k}
