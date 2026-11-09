"""Rewritten program with Pollard-Brent factorization and larger search space."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 135  # safety margin before 150s kill

    # Precompute small primes up to 10^6 for fast trial division
    SMALL_LIMIT = 10**6
    sieve = bytearray(b'\x01') * (SMALL_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(SMALL_LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, SMALL_LIMIT + 1) if sieve[i]]

    def is_prime_mr(n):
        """Deterministic Miller-Rabin for n < 3.3 * 10^24."""
        if n < 2:
            return False
        # Check small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 = d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Witnesses sufficient for n < 3.3 * 10^24
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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

    def pollard_brent(n):
        """Pollard's Brent cycle detection factorization."""
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if is_prime_mr(n):
            return n
        random.seed(n % (2**32))
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g = r = q = 1
            x = ys = y
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

    def factorize(n):
        """Return list of prime factors of n."""
        factors = []
        # Trial division by small primes
        for p in small_primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1:
            # n might be prime or a product of two large primes
            if is_prime_mr(n):
                factors.append(n)
            else:
                # Use Pollard-Brent
                stack = [n]
                while stack:
                    x = stack.pop()
                    if x == 1:
                        continue
                    if is_prime_mr(x):
                        factors.append(x)
                    else:
                        d = pollard_brent(x)
                        stack.append(d)
                        stack.append(x // d)
        return factors

    def largest_prime_factor(n):
        """Find the largest prime factor of n."""
        if n <= 1:
            return 1
        factors = factorize(n)
        return max(factors)

    def max_run_for_prime(p):
        """Find the maximum k such that p^2+1, ..., p^2+k all have prime factors <= p."""
        base = p * p
        k = 0
        while True:
            if time.time() > deadline - 5:
                break
            val = base + k + 1
            lpf = largest_prime_factor(val)
            if lpf > p:
                break
            k += 1
            if k >= 200:
                break
        return k

    # Search strategy: try many primes, focusing on larger ones where longer runs might exist
    # But also try some smaller primes in case they have surprisingly long runs
    
    best_k = 0
    best_p = 2

    # First pass: try primes up to 100000
    # Generate primes using sieve
    SIEVE_LIMIT = 100000
    sieve2 = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve2[0] = sieve2[1] = 0
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if sieve2[i]:
            sieve2[i*i::SIEVE_LIMIT+1]  # placeholder
    # Proper sieve
    sieve2 = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve2[0] = sieve2[1] = 0
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if sieve2[i]:
            for j in range(i*i, SIEVE_LIMIT+1, i):
                sieve2[j] = 0
    primes_small = [i for i in range(2, SIEVE_LIMIT+1) if sieve2[i]]

    for p in primes_small:
        if time.time() > deadline - 10:
            break
        k = max_run_for_prime(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Second pass: try larger primes in random or structured fashion
    # Focus on primes in ranges where longer runs might be found
    # Try primes up to 10^7 in chunks
    for start in range(100000, 10**7, 100000):
        if time.time() > deadline - 15:
            break
        end = min(start + 100000, 10**7)
        # Sieve this range
        chunk = bytearray(b'\x01') * (end - start)
        for p in primes_small:
            if p * p > end:
                break
            # Find first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            if first < p * p:
                first = p * p
            for j in range(first, end, p):
                chunk[j - start] = 0
        for i in range(start, end):
            if time.time() > deadline - 15:
                break
            if chunk[i - start] and i >= 2:
                # i is prime (since we've divided by all primes up to sqrt(end))
                k = max_run_for_prime(i)
                if k > best_k:
                    best_k = k
                    best_p = i

    # If we still have time, try some very large primes randomly
    while time.time() < deadline - 20:
        # Try random primes in range [10^7, 10^9]
        # Generate a random candidate and test if prime
        for _ in range(100):
            if time.time() > deadline - 20:
                break
            candidate = random.randint(10**7, 10**9)
            if candidate % 2 == 0:
                candidate += 1
            # Make it odd
            if candidate % 2 == 0:
                candidate += 1
            if is_prime_mr(candidate):
                k = max_run_for_prime(candidate)
                if k > best_k:
                    best_k = k
                    best_p = candidate

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
