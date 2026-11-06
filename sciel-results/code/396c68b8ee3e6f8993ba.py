"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def is_prime_mr(n):
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
    # Deterministic bases for n < 2^64
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
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g

def factorize(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime_mr(n):
        factors.append(n)
        return factors
    d = pollard_brent(n)
    factorize(d, factors)
    factorize(n // d, factors)
    return factors

def largest_prime_factor(n):
    if n == 1:
        return 1
    if is_prime_mr(n):
        return n
    d = pollard_brent(n)
    return max(largest_prime_factor(d), largest_prime_factor(n // d))

def is_p_smooth(n, p):
    """Check if all prime factors of n are <= p."""
    if n == 1:
        return True
    if is_prime_mr(n):
        return n <= p
    d = pollard_brent(n)
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def sieve_primes(limit):
    if limit < 2:
        return []
    is_p = [True] * (limit + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, limit + 1, i):
                is_p[j] = False
    return [i for i, v in enumerate(is_p) if v]

def construct():
    # Search for prime p that maximizes k
    best_k = 0
    best_p = 2
    
    # Check small primes first with trial division
    small_primes = sieve_primes(100000)
    
    for p in small_primes:
        if p > 10**9:
            break
        k = 0
        for i in range(1, 201):
            val = p * p + i
            if not is_p_smooth(val, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
