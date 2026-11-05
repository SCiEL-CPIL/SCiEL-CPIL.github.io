import math
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135.0  # Safety margin before 150s limit

    # Sieve primes up to 10^6
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit_p + 1, i):
                is_prime[j] = 0

    primes = [i for i in range(2, limit_p + 1) if is_prime[i]]

    def is_prime_mr(n):
        """Deterministic Miller-Rabin for n < 3.3e24"""
        if n < 2:
            return False
        # Small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        # Write n-1 = d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Bases sufficient for n < 3.3e24
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    def phi(x):
        """Compute Euler's totient of x using precomputed primes + Miller-Rabin for large factors."""
        if x == 1:
            return 1
        result = x
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            result -= result // n
        return result

    def find_run_start(s, max_k=200):
        """Find the length of the run starting at s where all phi values are distinct."""
        seen = set()
        k = 0
        for i in range(max_k):
            n = s + i
            if n > 10**12:
                break
            v = phi(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    best_k = 1
    best_s = 1

    # Strategy: random sampling of large numbers + local search
    random.seed(42)

    # Phase 1: Random sampling
    while time.time() < DEADLINE - 30:
        # Sample in log-uniform distribution from 10^6 to 10^12
        log_lo = math.log10(10**6)
        log_hi = math.log10(10**12)
        log_s = random.uniform(log_lo, log_hi)
        s = int(10**log_s)
        s = max(1, min(s, 10**12 - 200))

        k = find_run_start(s)
        if k > best_k:
            best_k = k
            best_s = s

    # Phase 2: Local search around best found
    if best_k > 1:
        # Try nearby starts
        for offset in range(-50, 51):
            if time.time() > DEADLINE - 5:
                break
            s = best_s + offset
            if s < 1 or s + 200 > 10**12 + 1:
                continue
            k = find_run_start(s)
            if k > best_k:
                best_k = k
                best_s = s

    # Phase 3: More random sampling with focus on promising regions
    remaining = DEADLINE - time.time()
    if remaining > 5:
        for _ in range(int(remaining * 50)):  # ~50 attempts per second
            if time.time() > DEADLINE - 2:
                break
            # Bias towards larger numbers where collisions are rarer
            log_s = random.uniform(math.log10(10**9), math.log10(10**12))
            s = int(10**log_s)
            s = max(1, min(s, 10**12 - 200))
            k = find_run_start(s)
            if k > best_k:
                best_k = k
                best_s = s
                # Local search around new best
                for offset in range(-10, 11):
                    if time.time() > DEADLINE - 2:
                        break
                    s2 = best_s + offset
                    if s2 < 1 or s2 + 200 > 10**12 + 1:
                        continue
                    k2 = find_run_start(s2)
                    if k2 > best_k:
                        best_k = k2
                        best_s = s2

    return {'start': best_s, 'length': best_k}
