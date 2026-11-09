import time
import random
import math

# Deterministic Miller-Rabin for n < 2^64
def is_prime(n):
    if n < 2:
        return False
    # Small primes
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
    # Bases for 64-bit
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = random.randrange(2, n)
        y = x
        c = random.randrange(1, n)
        d = 1
        q = 1
        r = 1
        m = 128
        ys = y
        while d == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and d == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                d = math.gcd(q, n)
                k += m
            r *= 2
        if d != n:
            return d
        d = 1
        while d == 1:
            ys = (ys * ys + c) % n
            d = math.gcd(abs(x - ys), n)
        if d != n:
            return d

def sieve_primes(limit):
    if limit < 2:
        return []
    is_p = [True] * (limit + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, limit + 1, i):
                is_p[j] = False
    return [i for i in range(2, limit + 1) if is_p[i]]

_SMALL_PRIMES = sieve_primes(10000)

def is_p_smooth(n, p, small_primes):
    if n <= 1:
        return True
    if n <= p:
        return True
    
    # Strip small factors
    for q in small_primes:
        if q > p:
            break
        if q > n:
            break
        while n % q == 0:
            n //= q
    
    if n == 1:
        return True
    if n <= p:
        return True
    
    # n > p, no small factors
    if is_prime(n):
        return False
        
    # n is composite
    d = pollard_rho(n)
    if d <= p:
        return is_p_smooth(n // d, p, small_primes)
    if n // d <= p:
        return is_p_smooth(d, p, small_primes)
    return is_p_smooth(d, p, small_primes) and is_p_smooth(n // d, p, small_primes)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_k = 0
    best_p = 2
    
    # Check p=2
    k = 0
    while k < 200 and is_p_smooth(4 + k + 1, 2, _SMALL_PRIMES):
        k += 1
    if k > best_k:
        best_k = k
        best_p = 2
        
    # Sieve primes up to 10^6
    primes = sieve_primes(1000000)
    
    for p in primes:
        if time.time() > deadline:
            break
        if p == 2:
            continue
            
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() > deadline:
                break
            if not is_p_smooth(p_sq + k + 1, p, _SMALL_PRIMES):
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
