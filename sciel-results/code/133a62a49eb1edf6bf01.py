import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    # Deterministic Miller-Rabin for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in MR_BASES:
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
        if n % 5 == 0:
            return 5
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = 128
            g, q, r = 1, 1, 1
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

    # Precompute primes up to 10^6 for trial division
    def sieve_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_p in enumerate(sieve) if is_p]

    TRIAL_DIV_LIMIT = 10**6
    trial_primes = sieve_primes(TRIAL_DIV_LIMIT)

    def is_p_smooth(n, p):
        """
        Returns True if all prime factors of n are <= p.
        Uses trial division by precomputed primes up to min(p, TRIAL_DIV_LIMIT),
        then Miller-Rabin / Pollard-Brent for the remainder.
        """
        if n <= 1:
            return True

        # Trial division by precomputed primes
        for q in trial_primes:
            if q > p:
                break
            if q * q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True

        # After trial division, n is either 1, prime, or has all prime factors > TRIAL_DIV_LIMIT
        if n == 1:
            return True

        # Check if n is prime
        if is_prime(n):
            return n <= p

        # n is composite with factors > TRIAL_DIV_LIMIT
        # Use Pollard-Brent to factor it
        def factorize_remaining(m, limit):
            if m <= 1:
                return True
            if m <= limit:
                return True
            if is_prime(m):
                return m <= limit
            d = pollard_brent(m)
            return factorize_remaining(d, limit) and factorize_remaining(m // d, limit)

        return factorize_remaining(n, p)

    def check_run(p, max_k=200):
        """
        Returns k such that p^2+1 ... p^2+k are p-smooth.
        """
        k = 0
        base = p * p
        while k < max_k:
            n = base + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        return k

    # Phase 1: Search primes up to 5 million
    limit = 5 * 10**6
    primes = sieve_primes(limit)

    best_k = 0
    best_p = 2

    for p in primes:
        if time.time() > deadline:
            break

        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Try larger primes
    if time.time() < deadline - 15:
        for _ in range(200):
            if time.time() > deadline:
                break
            x = random.randrange(10**7, 5 * 10**8)
            if x % 2 == 0:
                x += 1
            while not is_prime(x):
                x += 2
                if x > 5 * 10**8:
                    break
            if x > 5 * 10**8:
                continue
            k = check_run(x)
            if k > best_k:
                best_k = k
                best_p = x

    return {'p': best_p, 'k': best_k}
