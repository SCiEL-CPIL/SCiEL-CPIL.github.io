import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # safety margin before 120s limit

    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Sieve primes up to 2e6 for trial division and candidate p values
    SIEVE_LIMIT = 2000000
    primes = sieve_primes(SIEVE_LIMIT)

    # Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
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

    # Pollard-Brent factorization
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y, c, m = 2, 1, 128
            g, r, q = 1, 0, 1
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
                r += 1
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g
            c += 1

    def factorize(n, primes):
        """Fully factor n using trial division then Pollard-Brent."""
        factors = []
        m = n
        for q in primes:
            if q * q > m:
                break
            while m % q == 0:
                factors.append(q)
                m //= q
        if m > 1:
            # m is either prime or product of large factors
            stack = [m]
            while stack:
                val = stack.pop()
                if val == 1:
                    continue
                if is_prime_mr(val):
                    factors.append(val)
                else:
                    d = pollard_brent(val)
                    stack.append(d)
                    stack.append(val // d)
        return factors

    def is_p_smooth(n, p, primes):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Trial division with precomputed primes
        m = n
        for q in primes:
            if q > p:
                break
            if q * q > m:
                break
            while m % q == 0:
                m //= q
        if m == 1:
            return True
        if m <= p:
            return True
        # m has a factor > p; check if m itself is prime > p
        if is_prime_mr(m):
            return False
        # m is composite; use Pollard-Brent to find a factor
        d = pollard_brent(m)
        return is_p_smooth(d, p, primes) and is_p_smooth(m // d, p, primes)

    def check_run(p, primes, deadline):
        """Check the run length starting from p^2+1."""
        k = 0
        p_sq = p * p
        for i in range(1, 201):
            if time.time() > deadline:
                break
            n = p_sq + i
            if is_p_smooth(n, p, primes):
                k += 1
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Phase 1: Exhaustive search over primes p <= SIEVE_LIMIT
    for p in primes:
        if time.time() > DEADLINE - 5:
            break
        if p < 3:
            continue
        k = check_run(p, primes, DEADLINE - 2)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: If time remains, search larger primes using segmented approach
    # For p > SIEVE_LIMIT, we need to check p-smoothness of p^2+i
    # where i is 1..200. The numbers p^2+i are ~p^2, so their factors
    # up to sqrt(p^2+i) ~ p need checking.
    # We can use a segmented sieve on [p^2+1, p^2+200] with primes <= p.
    # But generating primes up to p for p up to 10^9 is expensive.
    # Instead, we use trial division with our precomputed primes (up to 2e6)
    # and Pollard-Brent for the remainder.

    # Generate candidate large primes to check
    # We'll check primes in the range [SIEVE_LIMIT, 10^9]
    # But we can't sieve up to 10^9. Instead, we'll check random-ish large primes
    # or use a segmented sieve for primality of candidates.

    # For now, if time permits, try some larger primes
    if time.time() < DEADLINE - 10:
        # Try primes just above SIEVE_LIMIT
        # Use a simple primality test to find primes in [SIEVE_LIMIT, SIEVE_LIMIT + 100000]
        start = SIEVE_LIMIT + 1
        if start % 2 == 0:
            start += 1
        for candidate in range(start, min(start + 100000, 10**9), 2):
            if time.time() > DEADLINE - 5:
                break
            if is_prime_mr(candidate):
                k = check_run(candidate, primes, DEADLINE - 2)
                if k > best_k:
                    best_k = k
                    best_p = candidate

    return {'p': best_p, 'k': best_k}
