import time
import math
import random

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 110.0 # 10s margin

# Pre-compute small primes for trial division
def sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

SMALL_PRIMES = sieve(10**5)

# Miller-Rabin Primality Test
def is_prime(n):
    if n < 2:
        return False
    # Small primes check
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
    # Deterministic bases for n < 3.3 * 10^24
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in bases:
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

# Pollard-Brent Factorization
def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, n - 1)
        g = 1
        r = 1
        q = 1
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
        # If g == n, retry with new parameters

# Check if n is p-smooth
def is_p_smooth(n, p):
    # Trial divide by small primes
    for q in SMALL_PRIMES:
        if q > p:
            break
        if q * q > n:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
            if n == 1:
                return True
    
    if n == 1:
        return True
        
    # If p is small, we've removed all factors <= p
    if p < 10**5:
        return False

    # n > 1, p >= 10^5
    if is_prime(n):
        return n <= p
    
    # n is composite
    d = pollard_brent(n)
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

# Generate primes up to limit
def generate_primes(limit):
    return sieve(limit)

def construct():
    global START_TIME
    best_k = 0
    best_p = 2
    
    # Search small primes first (cheap)
    primes = generate_primes(10**6)
    
    for p in primes:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        k = 0
        p2 = p * p
        while k < 200:
            n = p2 + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
    # If time remains, search random larger primes
    if time.time() - START_TIME < TIME_LIMIT - 5.0:
        # Random search in [10^6, 10^9]
        # We need to find primes.
        # Simple method: pick random odd number, check primality.
        attempts = 0
        while time.time() - START_TIME < TIME_LIMIT - 1.0 and attempts < 5000:
            attempts += 1
            p = random.randint(10**6, 10**9)
            if p % 2 == 0:
                p += 1
            if not is_prime(p):
                continue
                
            k = 0
            p2 = p * p
            while k < 200:
                n = p2 + k + 1
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
                    
            if k > best_k:
                best_k = k
                best_p = p
                
    return {'p': best_p, 'k': best_k}
