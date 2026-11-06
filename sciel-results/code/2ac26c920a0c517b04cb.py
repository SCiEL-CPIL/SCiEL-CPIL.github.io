import math
import time
import random

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
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

    def all_factors_leq(v, limit):
        if v <= 1:
            return True
        if is_prime(v):
            return v <= limit
        n = v
        # Trial division by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
        for p in small_primes:
            if p > limit:
                break
            while n % p == 0:
                n //= p
        if n == 1:
            return True
        if is_prime(n):
            return n <= limit
        # Pollard's Rho for remaining composite
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                if m > limit:
                    return False
            else:
                d = pollard_brent(m)
                if d == m:
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return True

    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                step = i
                start = i * i
                is_p[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        return [i for i in range(2, limit+1) if is_p[i]]

    best_k = 0
    best_p = 2

    # Generate candidates
    # 1. Sieve up to 10^6
    primes_small = sieve(10**6)
    
    # 2. Add some random large primes
    random_primes = []
    for _ in range(500):
        if time.time() > deadline - 30:
            break
        # Generate random odd number in [10^6, 10^9]
        start = 10**6
        end = 10**9
        n = random.randint(start, end)
        if n % 2 == 0:
            n += 1
        while n < end:
            if is_prime(n):
                random_primes.append(n)
                break
            n += 2

    candidates = list(set(primes_small + random_primes))
    # Sort descending: larger p is generally better for smoothness
    candidates.sort(reverse=True)

    for p in candidates:
        if time.time() > deadline - 5:
            break
        if p > 10**9:
            continue
        k = 0
        while k < 200:
            if time.time() > deadline - 5:
                break
            v = p * p + k + 1
            if all_factors_leq(v, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a very good k, we can potentially stop or continue
            # But let's continue to see if we can find better

    return {'p': best_p, 'k': best_k}
