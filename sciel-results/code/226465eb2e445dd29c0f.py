import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # safety margin before 120s limit

    # Deterministic Miller-Rabin for n < 3,474,749,660,383
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for a in [2, 3, 5, 7, 11, 13, 17]:
            if a >= n:
                continue
            d = n - 1
            r = 0
            while d % 2 == 0:
                d //= 2
                r += 1
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

    # Pollard's Rho algorithm for finding a factor
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

    # Get largest prime factor of n
    def largest_prime_factor(n):
        if n <= 1:
            return n
        # Try small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return max(p, largest_prime_factor(n // p))
        if is_prime(n):
            return n
        # Use Pollard's Rho
        d = pollard_rho(n)
        return max(largest_prime_factor(d), largest_prime_factor(n // d))

    # Check if all numbers from p^2+1 to p^2+k have all prime factors <= p
    def check_run(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            if time.time() > DEADLINE - 5:
                break
            n = p_sq + i
            if largest_prime_factor(n) <= p:
                k += 1
            else:
                break
        return k

    # Generate primes up to some limit using sieve
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    best_k = 0
    best_p = 2

    # First, search small primes with full k range
    small_primes = sieve_primes(10000)
    for p in small_primes:
        if time.time() > DEADLINE - 10:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Then search larger primes with reduced k range
    # For larger p, we can only check up to a smaller k before timeout
    large_prime_ranges = [
        (10000, 50000),
        (50000, 200000),
        (200000, 1000000),
        (1000000, 5000000),
        (5000000, 20000000),
        (20000000, 100000000),
    ]

    for lo, hi in large_prime_ranges:
        if time.time() > DEADLINE - 15:
            break
        # Use a coarser sieve for larger ranges
        step = max(1, (hi - lo) // 100000)  # sample at most ~100k candidates
        candidates = []
        for n in range(lo, hi + 1, step):
            if is_prime(n):
                candidates.append(n)
        
        for p in candidates:
            if time.time() > DEADLINE - 10:
                break
            # For larger p, limit the k check
            max_k_check = min(200, max(1, int((DEADLINE - time.time()) / 0.05)))
            k = check_run(p, k_max=max_k_check)
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
