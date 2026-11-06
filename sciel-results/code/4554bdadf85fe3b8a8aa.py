import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    # Deterministic Miller-Rabin for n < 2^64
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n == 2 or n == 3
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Witnesses sufficient for n < 3.4 * 10^14
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
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

    def largest_prime_factor(n):
        if n <= 1:
            return 1
        if is_prime(n):
            return n
        # Trial division by small primes (up to 1000)
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                if n == 1:
                    return p
                if is_prime(n):
                    return n
        # If n is still large, use Pollard-Brent
        factors = []
        while n > 1:
            if is_prime(n):
                factors.append(n)
                break
            d = pollard_brent(n)
            if d == n:
                # Retry
                continue
            factors.append(d)
            n //= d
        return max(factors)

    # Pre-compute small primes for trial division speedup in the main loop
    # We'll use a simple sieve up to 10000 for initial filtering
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]

    # Helper to check if n has any prime factor > p
    def has_large_factor(n, p):
        if n <= 1:
            return False
        if p < 2:
            return True
        # Trial divide by small primes
        for sp in small_primes:
            if sp > p:
                break
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return False
        if n == 1:
            return False
        # n is now composed of primes > 10000 (or > p if p < 10000)
        # If n is prime, check if n <= p
        if is_prime(n):
            return n > p
        # n is composite. Find its largest prime factor.
        return largest_prime_factor(n) > p

    best_k = 0
    best_p = 2

    # Search strategy:
    # 1. Check small primes (up to 10000) quickly
    # 2. Check random primes in various ranges
    # 3. Focus on ranges where long runs are more likely

    # First, check small primes
    for p in small_primes:
        if p > 10000:
            break
        p2 = p * p
        k = 0
        while k < 200:
            val = p2 + k + 1
            if has_large_factor(val, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p

    # Now search random primes in larger ranges
    # We'll try different ranges to maximize coverage
    ranges = [
        (10**4, 10**5),
        (10**5, 10**6),
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9),
        (10**9, 10**10)
    ]

    # Weight ranges by expected value
    # Larger p means p^2 is larger, so p^2+i has more potential factors
    # But also, the smoothness condition is harder for larger p
    # Let's try a mix

    while time.time() - start_time < DEADLINE:
        # Pick a range
        r = random.choice(ranges)
        lo, hi = r

        # Generate a random prime in this range
        found_prime = False
        for _ in range(50):
            p = random.randint(lo, hi)
            p |= 1  # Make odd
            if is_prime(p):
                found_prime = True
                break

        if not found_prime:
            continue

        # Check runs for this p
        p2 = p * p
        k = 0
        while k < 200:
            val = p2 + k + 1
            if has_large_factor(val, p):
                break
            k += 1

        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
