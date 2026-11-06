import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 115.0

    # Miller-Rabin primality test
    def is_prime(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Test bases
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

    # Pollard's rho
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

    def factor(n, factors=None):
        if factors is None:
            factors = {}
        if n == 1:
            return factors
        if is_prime(n):
            factors[n] = factors.get(n, 0) + 1
            return factors
        d = pollard_rho(n)
        factor(d, factors)
        factor(n // d, factors)
        return factors

    def phi_large(n):
        if n == 1:
            return 1
        facs = factor(n)
        result = n
        for p in facs:
            result -= result // p
        return result

    # Precompute phi for small numbers using Sieve
    SIEVE_LIMIT = 10**6
    phi_small = list(range(SIEVE_LIMIT + 1))
    for i in range(2, SIEVE_LIMIT + 1):
        if phi_small[i] == i:  # i is prime
            for j in range(i, SIEVE_LIMIT + 1, i):
                phi_small[j] -= phi_small[j] // i

    best_k = 1
    best_s = 1

    # Phase 1: Dense scan of small numbers
    # We can check windows of size up to 200
    # Scanning 10^6 numbers is fast
    for s in range(1, SIEVE_LIMIT - 200):
        if time.time() - start_time > DEADLINE:
            break
        seen = set()
        k = 0
        while k < 200:
            v = phi_small[s + k]
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s

    # Phase 2: Random search for larger numbers
    random.seed(42)
    max_n = 10**12

    def check_window(s, max_len=200):
        seen = set()
        k = 0
        while k < max_len and s + k <= max_n:
            if s + k <= SIEVE_LIMIT:
                v = phi_small[s + k]
            else:
                v = phi_large(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    # Search in the large range
    while time.time() - start_time < DEADLINE:
        s = random.randint(SIEVE_LIMIT, max_n - 200)
        k = check_window(s)
        if k > best_k:
            best_k = k
            best_s = s
            # If we find a good one, do local refinement
            if k > 10:
                for offset in range(-50, 51, 5):
                    if time.time() - start_time > DEADLINE:
                        break
                    ns = s + offset
                    if 1 <= ns <= max_n - 200:
                        nk = check_window(ns)
                        if nk > best_k:
                            best_k = nk
                            best_s = ns

    return {'start': best_s, 'length': best_k}
