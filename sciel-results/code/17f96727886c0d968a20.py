import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 110  # seconds, safety margin

    def is_prime_mr(n):
        """Deterministic Miller-Rabin for n < 2^64"""
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0:
            return False
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def pollard_brent(n):
        """Pollard's rho with Brent's improvement"""
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g = r = q_val = 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q_val = q_val * abs(x - y) % n
                    g = _gcd(q_val, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = _gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    def sieve_primes(limit):
        """Sieve of Eratosthenes up to limit using bytearray for speed."""
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Precompute small primes up to 10^6 for fast trial division
    SMALL_LIMIT = 10**6
    small_primes = sieve_primes(SMALL_LIMIT)

    def is_p_smooth_fast(n, p):
        """Check if all prime factors of n are <= p using precomputed primes."""
        if n <= 1:
            return True
        # Divide out small prime factors up to min(p, sqrt(n))
        for q in small_primes:
            if q > p or q * q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
        if n == 1:
            return True
        # n is now prime or composite with factors > SMALL_LIMIT
        if is_prime_mr(n):
            return n <= p
        # Composite: factor with Pollard-Brent and recurse
        f = pollard_brent(n)
        return is_p_smooth_fast(f, p) and is_p_smooth_fast(n // f, p)

    best_k = 0
    best_p = 2

    # Search primes in increasing order
    p = 3
    while p <= 1000000000:
        if time.time() - start_time > DEADLINE:
            break
        if is_prime_mr(p):
            base = p * p
            k = 0
            while k < 200:
                if time.time() - start_time > DEADLINE:
                    break
                if is_p_smooth_fast(base + k + 1, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
                if k >= 200:
                    break
        p += 2

    return {'p': best_p, 'k': best_k}
