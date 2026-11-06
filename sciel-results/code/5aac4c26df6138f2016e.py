import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 110
    
    # Miller-Rabin
    def is_prime(n):
        if n < 2: return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Brent
    def pollard_brent(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        y, c, m = random.randrange(1, n), random.randrange(1, n), random.randrange(1, n)
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
                if g > 1: break
        return g

    def is_p_smooth(n, p):
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        
        # Trial division for small factors
        f = None
        for d in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % d == 0:
                f = d
                break
        
        if f is None:
            f = pollard_brent(n)
            
        # Check if f is p-smooth and n/f is p-smooth
        # Optimization: if f > p and is_prime(f), return False
        if f > p and is_prime(f):
            return False
            
        # Recurse
        return is_p_smooth(f, p) and is_p_smooth(n // f, p)

    best_p = 0
    best_k = 0
    
    # Search primes in [10^6, 10^9]
    # We need to generate primes. 
    # Random sampling: pick random x, find next prime.
    
    # To speed up, we can pre-filter with small primes.
    
    while time.time() - start_time < TIME_LIMIT:
        # Pick random base
        base = random.randint(10**6, 10**9)
        # Find next prime
        p = base
        while not is_prime(p):
            p += 1
            if p > 10**9:
                break
        
        if p > 10**9:
            continue
            
        k = 0
        base_sq = p * p
        while k < 200:
            if time.time() - start_time > TIME_LIMIT:
                break
            n = base_sq + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
