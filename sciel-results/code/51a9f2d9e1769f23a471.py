import math
import time
import numpy as np

def construct():
    # 1. Sieve of Eratosthenes using NumPy for speed
    def get_primes(n):
        if n < 2:
            return []
        sieve = np.ones(n + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(np.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(10**6)
    small_primes = [p for p in primes if p < 1000]

    # 2. Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
        for a in bases:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 3. Pollard's Rho with Brent's cycle detection
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
        # Use deterministic c values to avoid random dependency or slow random gen
        # Or just use a fixed seed if needed, but standard random is fine.
        import random
        while True:
            y = random.randrange(2, n)
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

    # 4. Check if n is p-smooth (all prime factors <= p)
    def is_p_smooth(n, p, small_primes):
        if n <= 1:
            return True
        if n > p:
            # If n is prime, it must be <= p
            if is_prime(n):
                return n <= p
            # If n is composite, we need to check factors
            # Trial division by small primes
            for q in small_primes:
                if q > p:
                    break
                if q * q > n:
                    break
                if n % q == 0:
                    while n % q == 0:
                        n //= q
                    if n == 1:
                        return True
            # If n > 1 after trial division
            if n == 1:
                return True
            if is_prime(n):
                return n <= p
            # Use Pollard Rho to find a factor
            d = pollard_rho(n)
            if d > p or n // d > p:
                return False
            # Recursively check factors
            return is_p_smooth(d, p, small_primes) and is_p_smooth(n // d, p, small_primes)
        else:
            # n <= p, so any prime factor is <= n <= p
            return True

    best_k = 0
    best_p = 2
    start_time = time.time()
    timeout = 115

    # Search primes in increasing order
    for p in primes:
        if time.time() - start_time > timeout:
            break
        
        k = 0
        for i in range(1, 201):
            n = p*p + i
            if is_p_smooth(n, p, small_primes):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}
