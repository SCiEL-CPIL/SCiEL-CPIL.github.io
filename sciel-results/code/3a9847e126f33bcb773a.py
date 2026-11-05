"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s limit

    # Precompute primes up to 10^6 for efficient factorization
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    if limit_p >= 1:
        is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = 0

    primes = [i for i in range(2, limit_p + 1) if is_prime[i]]

    # Miller-Rabin for large numbers
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
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

    phi_cache = {}

    def phi(x):
        if x in phi_cache:
            return phi_cache[x]
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        phi_cache[x] = result
        return result

    def find_run_length(s, max_k=200):
        seen = set()
        k = 0
        for i in range(max_k):
            if time.time() > DEADLINE - 3:
                break
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    best_k = 1
    best_s = 1

    # Phase 1: Scan small range for baseline
    small_limit = 50000
    for s in range(1, min(small_limit - 200, 10**4)):
        if time.time() > DEADLINE - 20:
            break
        k = find_run_length(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
        if best_k >= 200:
            break

    if time.time() > DEADLINE - 15:
        return {'start': best_s, 'length': best_k}

    # Phase 2: Targeted search in large ranges
    random.seed(42)
    num_attempts = 0
    while time.time() < DEADLINE - 10:
        num_attempts += 1

        # Focus on large numbers where collisions are rarer
        if num_attempts % 4 == 0:
            s = random.randint(10**11, 10**12 - 200)
        elif num_attempts % 4 == 1:
            s = random.randint(10**10, 10**11)
        elif num_attempts % 4 == 2:
            s = random.randint(10**9, 10**10)
        else:
            s = random.randint(10**8, 10**9)

        k = find_run_length(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 200:
                break

    # Phase 3: Local optimization around best start
    if best_k < 200 and time.time() < DEADLINE - 5:
        for offset in range(-100, 101):
            if time.time() > DEADLINE - 3:
                break
            s = best_s + offset
            if s < 1 or s + 200 > 10**12:
                continue
            k = find_run_length(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 200:
                    break

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
