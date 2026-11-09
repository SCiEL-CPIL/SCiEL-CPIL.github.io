"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 140

    # Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n in (2, 3)
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # Pollard's rho (Brent's variant)
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
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

    # Full factorization
    def factorize(n):
        if n <= 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_brent(n)
        return factorize(d) + factorize(n // d)

    # Check if all prime factors of n are <= p
    def all_factors_at_most_p(n, p):
        if n <= 1:
            return True
        if n <= p:
            return True
        factors = factorize(n)
        return all(f <= p for f in factors)

    # Sieve primes up to limit
    def sieve_primes(limit):
        if limit < 2:
            return []
        s = bytearray(b'\x01') * (limit + 1)
        s[0] = s[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if s[i]:
                s[i*i::i] = bytearray(len(s[i*i::i]))
        return [i for i in range(2, limit + 1) if s[i]]

    # Precompute small primes for trial division
    SMALL_LIMIT = 200000
    small_primes = sieve_primes(SMALL_LIMIT)

    # For a given p, compute the maximum k such that p^2+1..p^2+k are all p-smooth
    def compute_k(p, max_k=200):
        p_sq = p * p
        k = 0
        for i in range(1, max_k + 1):
            n = p_sq + i
            m = n
            for sp in small_primes:
                if sp > p:
                    break
                if sp * sp > m:
                    break
                while m % sp == 0:
                    m //= sp
            if m == 1:
                k += 1
            elif m <= p:
                k += 1
            else:
                if is_prime_mr(m):
                    break
                else:
                    if not all_factors_at_most_p(m, p):
                        break
                    k += 1
        return k

    best_k = 0
    best_p = 2

    # Phase 1: Exhaustive search over small primes (up to 50000)
    primes_to_check = sieve_primes(50000)
    for p in primes_to_check:
        if time.time() > deadline:
            break
        k = compute_k(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search medium primes (50000 to 500000) in chunks
    if time.time() < deadline:
        for start in range(50000, 500000, 10000):
            if time.time() > deadline:
                break
            chunk_primes = sieve_primes(start + 10000)
            for p in chunk_primes:
                if p < start:
                    continue
                if time.time() > deadline:
                    break
                k = compute_k(p)
                if k > best_k:
                    best_k = k
                    best_p = p

    # Phase 3: Random sampling of larger primes
    if time.time() < deadline:
        attempts = 0
        while time.time() < deadline and attempts < 2000:
            attempts += 1
            if random.random() < 0.3:
                candidate = random.randint(500000, 2000000)
            elif random.random() < 0.5:
                candidate = random.randint(2000000, 10000000)
            else:
                candidate = random.randint(10000000, 100000000)
            
            if candidate % 2 == 0:
                candidate += 1
            
            found = False
            for _ in range(50):
                if is_prime_mr(candidate):
                    found = True
                    break
                candidate += 2
            if not found:
                continue
            
            k = compute_k(candidate)
            if k > best_k:
                best_k = k
                best_p = candidate

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
