import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Miller-Rabin Primality Test
    def is_prime(n):
        if n < 2:
            return False
        # Small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Witnesses for 32-bit/64-bit integers
        # Deterministic for n < 3,317,044,064,679,887,385,961,981
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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

    # Pollard's Rho
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        if n == 1:
            return []
        if is_prime(n):
            return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)

    def phi(n):
        if n == 1:
            return 1
        factors = factorize(n)
        unique_factors = set(factors)
        result = n
        for p in unique_factors:
            result -= result // p
        return result

    def check_run(s):
        # Check if phi(s), ..., phi(s+k-1) are distinct
        seen = set()
        k = 0
        # Max k is 200
        while k < 200:
            val = phi(s + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    best_k = 1
    best_s = 1

    # 1. Deterministic search in small range (faster than random for small s)
    # We can go up to 1,000,000 quickly
    # But phi is expensive. Let's do up to 100,000 first.
    limit_small = 100000
    for s in range(1, limit_small + 1):
        if time.time() > DEADLINE:
            break
        k = check_run(s)
        if k > best_k:
            best_k = k
            best_s = s

    # 2. Random search in larger range
    # We want to find longer runs.
    # Sample random s in [10^6, 10^12]
    # We need to be careful with time.
    # Each check_run takes ~k * cost(phi).
    # cost(phi) for 10^12 is ~1-2ms?
    # If k is small (say 5), it's fast.
    # If k is large (200), it's slower.
    
    random.seed(42) # For reproducibility in dev, maybe remove for eval? 
    # Actually, random sampling is good.
    
    # Let's try a few "structured" starts too?
    # No, random is safer.
    
    num_random_trials = 0
    while time.time() < DEADLINE:
        # Pick a random start
        # Bias towards larger numbers?
        # Uniform in [10^6, 10^12]
        s = random.randint(10**6, 10**12 - 200)
        
        # Check this run
        k = check_run(s)
        
        if k > best_k:
            best_k = k
            best_s = s
            
        # Optional: If we find a good run, search neighbors?
        # If k is large, maybe nearby s also have large k?
        # Not necessarily, but worth checking a few neighbors if time permits.
        if k >= best_k - 2:
             # Check a few neighbors
             for delta in [-1, 1, -2, 2]:
                 if time.time() > DEADLINE: break
                 s2 = s + delta
                 if s2 >= 1 and s2 + 199 <= 10**12:
                     k2 = check_run(s2)
                     if k2 > best_k:
                         best_k = k2
                         best_s = s2

        num_random_trials += 1
        # Throttle if phi is too slow?
        # If 1000 trials in 10s, that's 100/s.
        # 115s -> 11500 trials. Should be enough to find better than 20000 range.

    return {'start': best_s, 'length': best_k}
