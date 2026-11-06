import time
import math
import random

def is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.3e24"""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
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

def sieve(limit):
    is_prime_flag = bytearray(b'\x01') * (limit + 1)
    is_prime_flag[0] = is_prime_flag[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_flag[i]:
            is_prime_flag[i*i::i] = bytearray(len(is_prime_flag[i*i::i]))
    return [i for i in range(limit + 1) if is_prime_flag[i]]

def largest_prime_factor(n, small_primes, deadline=None):
    """Find the largest prime factor of n. Returns None if deadline exceeded."""
    if n <= 1:
        return 1
    
    # Try small primes first
    m = n
    largest = 1
    for p in small_primes:
        if p * p > m:
            break
        if m % p == 0:
            largest = p
            while m % p == 0:
                m //= p
    
    if m == 1:
        return largest
    
    if m <= small_primes[-1] if small_primes else True:
        # m is either prime or has all factors in small_primes
        if is_prime_mr(m):
            return max(largest, m)
        else:
            # m is composite with factors all <= small_primes[-1]
            # Find largest factor of m
            lm = m
            for p in small_primes:
                if p * p > lm:
                    break
                if lm % p == 0:
                    while lm % p == 0:
                        lm //= p
            if lm > 1:
                return max(largest, lm)
            return largest
    
    # m has a prime factor > small_primes[-1]
    # Check if m is prime
    if is_prime_mr(m):
        return max(largest, m)
    
    # Use Pollard's rho to find a factor
    d = pollard_brent(m)
    if d == m:
        return max(largest, m)
    
    # Recursively find largest prime factor of d and m//d
    lpf_d = largest_prime_factor(d, small_primes, deadline)
    lpf_m = largest_prime_factor(m // d, small_primes, deadline)
    return max(lpf_d, lpf_m)

def check_run(p, small_primes, deadline):
    """Check how many consecutive integers p^2+1, p^2+2, ... are p-smooth."""
    k = 0
    p2 = p * p
    while k < 200:
        if time.time() > deadline:
            break
        n = p2 + k + 1
        if n <= p:
            k += 1
            continue
        lpf = largest_prime_factor(n, small_primes, deadline)
        if lpf is None or lpf > p:
            break
        k += 1
    return k

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Sieve primes up to 10^6 for trial division and candidate generation
    SIEVE_LIMIT = 10**6
    small_primes = sieve(SIEVE_LIMIT)
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Check all primes up to 10^6
    for p in small_primes:
        if time.time() > deadline:
            break
        k = check_run(p, small_primes, deadline)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: If time remains, check larger primes up to 10^9
    # Generate primes in larger ranges
    if time.time() < deadline - 20:
        # Check primes in [10^6, 10^7]
        limit2 = 10**7
        if limit2 > SIEVE_LIMIT:
            primes2 = sieve(limit2)
            for p in primes2:
                if p <= SIEVE_LIMIT:
                    continue
                if time.time() > deadline:
                    break
                k = check_run(p, small_primes, deadline)
                if k > best_k:
                    best_k = k
                    best_p = p
    
    # Phase 3: If time remains, sample larger primes
    if time.time() < deadline - 10:
        # Try some random larger primes
        random.seed(42)
        for _ in range(50000):
            if time.time() > deadline:
                break
            p = random.randint(10**7, 10**9)
            if p % 2 == 0:
                p += 1
            # Quick check: is p prime?
            if not is_prime_mr(p):
                continue
            # Quick check: is p^2+1 prime? If so, k=0
            n1 = p * p + 1
            if is_prime_mr(n1) and n1 > p:
                continue
            k = check_run(p, small_primes, deadline)
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
