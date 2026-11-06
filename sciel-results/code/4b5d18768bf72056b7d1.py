import math
import time
import random

def is_prime_mr(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Small primes check
    for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    # Miller-Rabin with deterministic bases for n < 3.2 * 10^18
    # Bases: 2, 3, 5, 7, 11, 13, 17
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in [2, 3, 5, 7, 11, 13, 17]:
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
    
    y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), 128
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
    return g

def factorize(n):
    factors = []
    while n > 1:
        if is_prime_mr(n):
            factors.append(n)
            break
        d = pollard_brent(n)
        factors.append(d)
        n //= d
    return factors

def largest_prime_factor(n):
    if n <= 1:
        return 1
    facs = factorize(n)
    return max(facs)

def construct():
    start_time = time.time()
    deadline = 115.0  # 5s safety margin
    
    # We want to maximize k.
    # Let's search primes in increasing order of "potential".
    # Larger p allows larger range, but the condition is harder?
    # Actually, for p to be good, p^2+1 ... p^2+k must be p-smooth.
    # Let's try a mix of small and large primes.
    # The seed found something for p < 2000.
    # Let's try to find better p in [2000, 10^6] first, then higher.
    
    # Helper to get primes in a range
    def primes_in_range(start, end):
        # Simple sieve for small ranges, trial division for large sparse
        # For this task, we can just iterate odd numbers and check primality
        if start % 2 == 0:
            start += 1
        for i in range(start, end + 1, 2):
            if is_prime_mr(i):
                yield i

    best_k = 0
    best_p = 2
    
    # Strategy:
    # 1. Check a few known good candidates or random primes in mid-range.
    # 2. Systematic search in chunks.
    
    # Let's do a random search first to find a good p, then local search around it?
    # Or just scan primes.
    
    # To be efficient, we can skip p if p^2+1 is not p-smooth.
    # But we need to check k=1, 2, ...
    
    # Let's scan primes from 3 to 10^5 first (fast)
    # Then random samples in [10^5, 10^8]
    
    # Phase 1: Small primes
    for p in primes_in_range(3, 10000):
        if time.time() > deadline:
            break
        k = 0
        while k < 200:
            n = p * p + k + 1
            if largest_prime_factor(n) <= p:
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Medium primes, step size 2? No, just odd numbers and check prime.
    # We can sample.
    # Let's sample primes in [10^4, 10^6]
    # Number of primes ~ 10^6 / ln(10^6) ~ 10^6 / 14 ~ 70000.
    # We can check a subset.
    
    # Let's try a deterministic scan with a step, checking primality.
    # Or just random primes.
    
    # Let's use a sieve for a specific range to get primes quickly?
    # Sieve of Eratosthenes for 10^6 is fast.
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    # Sieve up to 10^6
    primes_1m = sieve(1000000)
    # Filter out those < 10000 already checked
    for p in primes_1m:
        if p < 10000:
            continue
        if time.time() > deadline:
            break
        k = 0
        while k < 200:
            n = p * p + k + 1
            if largest_prime_factor(n) <= p:
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 3: Larger primes, random sampling
    # p up to 10^9
    # We can't sieve. Use is_prime_mr.
    # Sample random odd numbers in [10^6, 10^9]
    
    # To make it deterministic, use a fixed seed or time-based?
    # Time-based is better for exploration, but reproducibility?
    # The task doesn't require reproducibility, just a good score.
    
    # Let's sample 5000 random primes in [10^6, 10^8]
    # and 500 in [10^8, 10^9]
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        p = random.randint(1000000, 100000000)
        if p % 2 == 0:
            p += 1
        # Find next prime
        while not is_prime_mr(p):
            p += 2
            if p > 100000000:
                break
        if p > 100000000:
            continue
            
        k = 0
        while k < 200:
            n = p * p + k + 1
            if largest_prime_factor(n) <= p:
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    for _ in range(500):
        if time.time() > deadline:
            break
        p = random.randint(100000000, 1000000000)
        if p % 2 == 0:
            p += 1
        while not is_prime_mr(p):
            p += 2
            if p > 1000000000:
                break
        if p > 1000000000:
            continue
            
        k = 0
        while k < 200:
            n = p * p + k + 1
            if largest_prime_factor(n) <= p:
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
