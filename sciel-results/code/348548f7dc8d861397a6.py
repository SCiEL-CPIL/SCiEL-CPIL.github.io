import math
import time
import random

def construct():
    deadline = time.time() + 113.0  # safety margin before 120s

    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    SIEVE_LIMIT = 100000
    primes = sieve(SIEVE_LIMIT)
    prime_set = set(primes)
    small_primes = [p for p in primes if p <= 1000]

    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        small_primes_mr = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for sp in small_primes_mr:
            if n == sp:
                return True
            if n % sp == 0:
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
        """Pollard's Brent cycle detection factorization."""
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

    def largest_prime_factor_leq(n, p, primes_list):
        """Return True if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Trial divide by primes <= p
        for q in primes_list:
            if q > p:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
        if n <= 1:
            return True
        if n <= p:
            return True
        # n > p: try to factor it further
        # If n is prime, it's a factor > p -> fail
        if is_prime_mr(n):
            return False
        # n is composite; try Pollard-Brent
        try:
            factor = pollard_brent(n)
            if factor > p:
                return False
            # Recursively check factor and n//factor
            if not largest_prime_factor_leq(factor, p, primes_list):
                return False
            if not largest_prime_factor_leq(n // factor, p, primes_list):
                return False
            return True
        except Exception:
            # If Pollard fails, be conservative
            return False

    def check_run(p):
        """Check how many consecutive numbers p^2+1 .. p^2+k are p-smooth."""
        k = 0
        p_sq = p * p
        for k_val in range(1, 201):
            v = p_sq + k_val
            if largest_prime_factor_leq(v, p, primes):
                k = k_val
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Check p = 2
    k = check_run(2)
    if k > best_k:
        best_k = k
        best_p = 2

    # DESCENDING search: large primes first
    # Start from the largest sieved prime and go down
    for p in reversed(primes):
        if time.time() > deadline:
            break
        if p < 2:
            continue
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
        # Early stop: if we've found a very good run, we can be more selective
        if best_k >= 150:
            break

    # Try larger primes beyond sieve using Miller-Rabin
    random.seed(42)
    checked = set(primes)
    # Search in descending order from large values
    for start in range(500000, 100000, -1000):
        if time.time() > deadline:
            break
        for candidate in range(start, start + 1000, 2):
            if time.time() > deadline:
                break
            if candidate > 10**9 or candidate < 100000:
                continue
            if candidate in checked:
                continue
            checked.add(candidate)
            if not is_prime_mr(candidate):
                continue
            p = candidate
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
        if best_k >= 150:
            break

    # Try even larger primes
    for _ in range(2000):
        if time.time() > deadline:
            break
        candidate = random.randint(1000000, 10000000)
        if candidate in checked:
            continue
        checked.add(candidate)
        if not is_prime_mr(candidate):
            continue
        p = candidate
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
