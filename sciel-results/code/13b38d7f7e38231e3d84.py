"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import sys

def is_prime_miller_rabin(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Bases sufficient for n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
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

def is_p_smooth(n, p, small_primes):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    # Trial division by small primes
    for q in small_primes:
        if q > p:
            # If q > p and n % q == 0, then n has a factor > p
            # But we break out of the loop if q*q > n as well
            if n % q == 0:
                return False
            break
        if q * q > n:
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    # Check if n is prime
    if is_prime_miller_rabin(n):
        return n <= p
    # n is composite
    # Find a factor
    d = pollard_brent(n)
    # If d > p, then n has a factor > p
    if d > p:
        return False
    # d <= p, so d is p-smooth. Check the rest.
    return is_p_smooth(n // d, p, small_primes)

def construct():
    # Generate primes up to 100000 for trial division
    limit = 100000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Exhaustive search for small primes
    # p^2 + 200 < 10^12 for p < 10^6, which is fast to factor
    for p in small_primes:
        if p > 1000000:
            break
        k = 0
        for j in range(1, 201):
            n = p * p + j
            if is_p_smooth(n, p, small_primes):
                k = j
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
        if k >= 200:
            break
            
    # Phase 2: Randomized search for larger primes
    # Sample random primes in [10^6, 10^9]
    # We need to be careful with time. 120s limit.
    # Let's try 1000 random samples.
    # Generating a random prime in [10^6, 10^9]
    
    # Helper to find a random prime in [low, high]
    def random_prime(low, high):
        while True:
            x = random.randrange(low, high)
            if x % 2 == 0:
                x += 1
            if is_prime_miller_rabin(x):
                return x

    # We will do a few hundred random checks
    # To avoid bias, we can also check a few specific large primes if we have time
    # But random is safer for finding outliers.
    
    num_samples = 500
    for _ in range(num_samples):
        p = random_prime(1000000, 1000000000)
        k = 0
        for j in range(1, 201):
            n = p * p + j
            if is_p_smooth(n, p, small_primes):
                k = j
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
        if k >= 200:
            break

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
