import time
import math
import random
import sys

# Sieve of Eratosthenes to get primes up to 10^6
def sieve(n):
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(n + 1) if is_prime[i]]

_SMALL_PRIMES = sieve(1000000)
_SMALL_PRIME_SET = set(_SMALL_PRIMES)

# Deterministic Miller-Rabin for 64-bit integers
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
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
    while True:
        y = random.randint(2, n - 1)
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

def factorize(n, primes):
    """
    Returns a list of prime factors of n.
    Uses trial division by small primes, then Pollard-Brent.
    """
    factors = []
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            while n % p == 0:
                factors.append(p)
                n //= p
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if is_prime(x):
            factors.append(x)
            continue
        d = pollard_brent(x)
        stack.append(d)
        stack.append(x // d)
    return factors

def is_p_smooth(n, p, primes):
    """
    Check if n is p-smooth using trial division + Pollard-Brent.
    """
    if n <= 1:
        return True
    
    # Trial division by small primes up to min(p, 10^6)
    limit = min(p, 1000000)
    for q in primes:
        if q > limit:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
    
    if n == 1:
        return True
    
    if is_prime(n):
        return n <= p
    
    # Pollard-Brent
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if is_prime(x):
            if x > p:
                return False
            continue
        
        r = int(math.isqrt(x))
        if r * r == x:
            stack.append(r)
            continue
        
        d = pollard_brent(x)
        stack.append(d)
        stack.append(x // d)
    
    return True

def check_run(p, primes):
    """
    Check the longest run of p-smooth numbers starting from p^2+1.
    """
    base = p * p
    k = 0
    for j in range(200):
        n = base + j + 1
        if is_p_smooth(n, p, primes):
            k = j + 1
        else:
            break
    return k

def construct():
    deadline = time.time() + 110
    best_k = 0
    best_p = 0
    
    # Check all small primes (p <= 10^6)
    for p in _SMALL_PRIMES:
        if time.time() > deadline:
            break
        
        # Quick filter: if p^2+1 is prime, it's > p, so not smooth
        # But we need to check smoothness anyway.
        k = check_run(p, _SMALL_PRIMES)
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we have time, try some larger primes
    if time.time() < deadline - 10:
        # Random search in [10^6, 10^9]
        while time.time() < deadline - 2:
            # Generate a random prime
            p = random.randint(10**6, 10**9)
            while not is_prime(p):
                p += 2
                if p > 10**9:
                    p = 10**6
            
            k = check_run(p, _SMALL_PRIMES)
            
            if k > best_k:
                best_k = k
                best_p = p
    
    if best_k < 1:
        best_p = 7
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
