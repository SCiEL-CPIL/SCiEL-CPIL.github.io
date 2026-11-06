import math
import random
import time

def is_prime(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
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
    """Find a non-trivial factor of n using Pollard's Brent algorithm."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        y = random.randint(2, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, n - 1)
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

def factorize(n):
    """Return a list of prime factors of n (with multiplicity)."""
    if n <= 1:
        return []
    if n == 2:
        return [2]
    if n % 2 == 0:
        return [2] + factorize(n // 2)
    if n % 3 == 0:
        return [3] + factorize(n // 3)
    if is_prime(n):
        return [n]
    factor = pollard_brent(n)
    return factorize(factor) + factorize(n // factor)

def largest_prime_factor(n):
    """Return the largest prime factor of n."""
    if n <= 1:
        return 1
    factors = factorize(n)
    return max(factors)

def is_p_smooth(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    if n <= p:
        return is_prime(n) or n == 1
    # Quick check: if n has a prime factor > p, it's not p-smooth
    # Use factorization to find the largest prime factor
    lpf = largest_prime_factor(n)
    return lpf <= p

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    best_p = 7
    best_k = 1
    
    # Search strategy:
    # 1. Try small primes first (fast to check, likely to have short runs)
    # 2. Then try larger primes in a structured way
    # For each prime p, check p^2+1, p^2+2, ... until we find a non-p-smooth number
    
    # Phase 1: Check primes up to 10000 quickly
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
                    211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293,
                    307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
                    401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
                    503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
                    601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691,
                    701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797,
                    809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887,
                    907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    
    for p in small_primes:
        if time.time() > deadline:
            break
        k = 0
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Check larger primes using a sieve up to a reasonable limit
    # For larger p, the numbers p^2+i are large, so factorization is expensive
    # But we can still try some
    
    # Generate primes up to 100000 using a simple sieve
    limit = 100000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Skip primes we already checked
    checked = set(small_primes)
    
    for p in primes:
        if time.time() > deadline:
            break
        if p < 1000 or p in checked:
            continue
        if p > limit:
            break
        
        k = 0
        # For larger p, we expect shorter runs, so limit the check
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
