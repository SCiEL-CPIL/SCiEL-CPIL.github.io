import time
import random
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Sieve of Eratosthenes for primes up to 10^6
    SIEVE_LIMIT = 10**6
    is_prime_arr = [True] * (SIEVE_LIMIT + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, SIEVE_LIMIT + 1, i):
                is_prime_arr[j] = False
    small_primes = [i for i, prime in enumerate(is_prime_arr) if prime]

    # Miller-Rabin for large numbers
    def is_prime_large(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
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

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            c = random.randint(1, n - 1)
            x = random.randint(2, n - 1)
            y = x
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factor_large(n, factors=None):
        if factors is None:
            factors = {}
        if n == 1:
            return factors
        if is_prime_large(n):
            factors[n] = factors.get(n, 0) + 1
            return factors
        d = pollard_rho(n)
        factor_large(d, factors)
        factor_large(n // d, factors)
        return factors

    @lru_cache(maxsize=None)
    def phi_fast(n):
        if n == 1:
            return 1
        result = n
        remaining = n
        # Trial division by small primes
        for p in small_primes:
            if p * p > remaining:
                break
            if remaining % p == 0:
                result -= result // p
                while remaining % p == 0:
                    remaining //= p
        if remaining > 1:
            # For n <= 10^12, if remaining > 1 after dividing by primes up to 10^6,
            # remaining MUST be prime.
            result -= result // remaining
        return result

    best_k = 1
    best_s = 1
    max_n = 10**12

    def check_window(s, max_len=200):
        seen = set()
        k = 0
        while k < max_len and s + k <= max_n:
            v = phi_fast(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    # Deterministic scan first
    scan_limit = 100000
    for s in range(1, scan_limit):
        if time.time() - start_time > DEADLINE:
            break
        k = check_window(s)
        if k > best_k:
            best_k = k
            best_s = s

    # Random search
    random.seed(42)
    while time.time() - start_time < DEADLINE:
        s = random.randint(1, max_n - 200)
        k = check_window(s)
        if k > best_k:
            best_k = k
            best_s = s
        if k > 5:
            # Local refinement with larger radius
            radius = 100
            for offset in range(-radius, radius + 1):
                if time.time() - start_time > DEADLINE:
                    break
                ns = s + offset
                if 1 <= ns <= max_n - 200:
                    nk = check_window(ns)
                    if nk > best_k:
                        best_k = nk
                        best_s = ns

    return {'start': best_s, 'length': best_k}
