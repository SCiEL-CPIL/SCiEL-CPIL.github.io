import time
import math
import random

# Sieve of Eratosthenes to get primes up to 10^7
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

_SMALL_PRIMES = sieve(10000000)
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

def is_p_smooth(n, p):
    """
    Check if all prime factors of n are <= p.
    """
    if n <= 1:
        return True
    
    # Trial division by small primes up to min(p, 10^7)
    limit = min(p, 10000000)
    for q in _SMALL_PRIMES:
        if q > limit:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
    
    if n == 1:
        return True
    
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
    
    # Otherwise, n is composite.
    # We need to check if all its prime factors are <= p.
    # Since we removed all factors <= min(p, 10^7), 
    # any remaining factor must be > 10^7 (or > p if p < 10^7).
    
    # If p < 10^7, then any remaining factor > p implies not smooth.
    # Wait, if p < 10^7, we divided by all q <= p. So any remaining factor > p.
    # So if p < 10^7 and n > 1, return False.
    if p < 10000000:
        return False
        
    # If p >= 10^7, remaining factors are > 10^7.
    # We need to factor n and check if all factors <= p.
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

def construct():
    deadline = time.time() + 115
    best_k = 0
    best_p = 0
    
    # Try primes in increasing order
    for p in _SMALL_PRIMES:
        if time.time() > deadline:
            break
        if p > 10**9:
            break
        
        # Check k sequentially
        # Quick fail: if p^2+1 is prime, it's not p-smooth
        n1 = p*p + 1
        if is_prime(n1):
            continue
            
        # Check i=1
        if not is_p_smooth(n1, p):
            continue
            
        k = 1
        # Check i=2..200
        for i in range(2, 201):
            n = p*p + i
            if not is_p_smooth(n, p):
                break
            k = i
            
        if k > best_k:
            best_k = k
            best_p = p
            
        if best_k >= 200:
            break
    
    # If we have time, try some larger primes
    if time.time() < deadline - 5:
        last_p = _SMALL_PRIMES[-1]
        while time.time() < deadline - 2:
            last_p += 2
            if last_p > 10**9:
                break
            if not is_prime(last_p):
                continue
            
            p = last_p
            n1 = p*p + 1
            if is_prime(n1):
                continue
            if not is_p_smooth(n1, p):
                continue
                
            k = 1
            for i in range(2, 201):
                n = p*p + i
                if not is_p_smooth(n, p):
                    break
                k = i
                
            if k > best_k:
                best_k = k
                best_p = p
                
            if best_k >= 200:
                break
    
    if best_k < 1:
        # Fallback: find any valid p,k
        # p=7, k=1 is valid (50 = 2*5^2, 5<=7)
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}
