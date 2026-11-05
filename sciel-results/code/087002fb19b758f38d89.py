import time
import random
import math

# Deterministic Miller-Rabin for 64-bit integers
def is_prime(n):
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    
    # Write n-1 as d * 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
        
    # Bases sufficient for n < 2^64
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for a in bases:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
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
    y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
    g, q, r, x, ys = 1, 1, 1, 0, y
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
    return g

def factorize(n):
    if n == 1:
        return []
    if is_prime(n):
        return [n]
    d = pollard_brent(n)
    return factorize(d) + factorize(n // d)

def is_p_smooth(n, p, small_primes):
    if n == 1:
        return True
    # Trial division by small primes
    for q in small_primes:
        if q > p:
            break
        if q * q > n:
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    # Check if remaining n is prime
    if is_prime(n):
        return n <= p
    # Factorize the rest
    factors = factorize(n)
    return all(f <= p for f in factors)

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    # Generate small primes for trial division
    # Limit to 10^6 is reasonable for speed/memory
    limit = 10**6
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    best_k = 0
    best_p = 3
    
    # Iterate through primes
    # We can generate primes on the fly or use a larger sieve
    # Let's use a larger sieve up to 10^7 if time permits, or just iterate
    # Iterating and checking primality is slower but flexible.
    # Let's use a sieve up to 10^7.
    limit2 = 10**7
    sieve2 = bytearray(b'\x01') * (limit2 + 1)
    sieve2[0:2] = b'\x00\x00'
    for i in range(2, int(limit2 ** 0.5) + 1):
        if sieve2[i]:
            sieve2[i*i::i] = bytearray(len(sieve2[i*i::i]))
            
    primes = [i for i, is_p in enumerate(sieve2) if is_p]
    
    for p in primes:
        if time.time() > deadline:
            break
        if p > 10**9:
            break
            
        k = 0
        # Check run length
        # Max k is 200
        while k < 200:
            n = p * p + k + 1
            # Optimization: if n is prime and n > p, it fails immediately
            # But is_prime is fast. is_p_smooth includes is_prime check.
            if not is_p_smooth(n, p, small_primes):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
