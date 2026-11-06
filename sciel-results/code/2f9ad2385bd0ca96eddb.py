import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Miller-Rabin primality test
    def is_prime(n):
        if n < 2:
            return False
        # Small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        # Write n-1 as d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Deterministic for n < 2^64
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
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

    def factor(n, res):
        if n == 1:
            return
        if is_prime(n):
            res.append(n)
            return
        d = pollard_rho(n)
        factor(d, res)
        factor(n // d, res)

    def is_p_smooth(n, p):
        # Check if all prime factors of n are <= p
        if n == 1:
            return True
        if n > p and is_prime(n):
            return False
        # Trial division by small primes first for speed
        # We can just factor and check max factor
        factors = []
        factor(n, factors)
        return max(factors) <= p

    # Sieve of Eratosthenes
    def sieve(limit):
        is_prime_arr = [True] * (limit + 1)
        is_prime_arr[0] = False
        is_prime_arr[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime_arr[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime_arr[j] = False
        return [i for i, v in enumerate(is_prime_arr) if v]

    best_k = 0
    best_p = 2
    
    # Start with small primes
    limit = 10000
    primes = sieve(limit)
    
    for p in primes:
        if time.time() > deadline:
            break
        k = 0
        # Check p^2 + 1 ... p^2 + 200
        # Optimize: if p^2 + j is prime, it fails.
        # If p^2 + j has a factor > p, it fails.
        
        # We can stop early if k reaches 200
        for j in range(1, 201):
            n = p * p + j
            # Quick check: if n is prime, fail
            if is_prime(n):
                break
            # Check if p-smooth
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p

    # If we have time, try larger primes
    # But factorization gets slower.
    # Let's try up to 100000 if time remains
    if time.time() < deadline - 20.0:
        limit2 = 100000
        primes2 = sieve(limit2)
        # Skip those already checked
        for p in primes2:
            if p < limit:
                continue
            if time.time() > deadline:
                break
            k = 0
            for j in range(1, 201):
                n = p * p + j
                if is_prime(n):
                    break
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
