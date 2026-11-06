import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # seconds; leave 10s margin before 150s kill
    
    # --- Prime helpers ---
    def is_prime_mr(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        # Deterministic for n < 3,317,044,064,679,887,385,961,981
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % a == 0: return n == a
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    def pollard_brent(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, n - 1)
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    def factorize(n):
        factors = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime_mr(m):
                factors.append(m)
                continue
            d = pollard_brent(m)
            stack.append(d)
            stack.append(m // d)
        return factors

    def largest_prime_factor(n):
        if n <= 1:
            return 1
        factors = factorize(n)
        return max(factors)

    # --- Sieve for small primes ---
    def sieve(n):
        """Return list of primes up to n."""
        if n < 2: return []
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, n + 1, i):
                    is_p[j] = False
        return [i for i in range(2, n + 1) if is_p[i]]

    # --- Main search ---
    best_k = 0
    best_p = 3

    def check_candidate(p):
        """Check p^2+1 ... p^2+k for p-smoothness. Return k."""
        k = 0
        for i in range(1, 201):
            val = p * p + i
            if largest_prime_factor(val) <= p:
                k = i
            else:
                break
        return k

    # Phase 1: Small primes via sieve (up to 200,000)
    small_primes = sieve(200_000)
    for p in small_primes:
        if p < 3:
            continue
        if time.time() - start_time > DEADLINE:
            break
        k = check_candidate(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random large primes
    random.seed(42)
    attempts = 0
    max_attempts = 50000
    while time.time() - start_time < DEADLINE and attempts < max_attempts:
        attempts += 1
        # Sample random numbers in range [10^7, 10^9]
        p = random.randint(10**7, 10**9)
        if p % 2 == 0:
            p += 1
        # Check primality
        if not is_prime_mr(p):
            continue
        k = check_candidate(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 3: Systematic search near known good primes
    # Search around primes that gave k > 0
    if best_k > 0:
        # Try neighbors of best_p
        for offset in range(1, 100):
            if time.time() - start_time > DEADLINE:
                break
            for p in [best_p + offset, best_p - offset]:
                if p < 3 or p > 10**9:
                    continue
                if not is_prime_mr(p):
                    continue
                k = check_candidate(p)
                if k > best_k:
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}
