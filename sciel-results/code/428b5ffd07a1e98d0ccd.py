"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 105.0  # safety margin before 150s kill

    # Precompute small primes up to 10^6 for trial division and sieve
    def sieve_primes(n):
        if n < 2:
            return []
        bs = bytearray(b'\x01') * (n + 1)
        bs[0] = bs[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if bs[i]:
                bs[i*i::i] = bytearray(len(bs[i*i::i]))
        return [i for i in range(2, n + 1) if bs[i]]

    small_primes = sieve_primes(10**6)
    small_primes_set = set(small_primes)

    # Deterministic Miller-Rabin for n < 3.2e18
    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
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

    # Pollard's rho with Brent's improvement
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            c = random.randrange(1, n - 1)
            y = random.randrange(0, n - 1)
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

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p"""
        if n <= 1:
            return True
        if is_prime(n):
            return n <= p

        # Trial division by small primes
        for sp in small_primes:
            if sp * sp > n:
                break
            if sp > p:
                break
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return True
                if not is_p_smooth(n, p):
                    return False
                return True

        if n == 1:
            return True

        if is_prime(n):
            return n <= p

        # Use Pollard's Rho to find a factor
        d = pollard_brent(n)
        if d == n:
            return n <= p

        if not is_p_smooth(d, p):
            return False
        if not is_p_smooth(n // d, p):
            return False
        return True

    def find_smooth_run(p, max_k=200):
        """
        Find the longest run of p-smooth numbers starting at p^2+1.
        Uses a segmented-sieve-style batched trial division to quickly
        identify numbers that are NOT p-smooth, avoiding expensive
        Pollard-Brent factorization for the vast majority of cases.
        """
        base = p * p
        low = base + 1
        high = base + max_k
        size = max_k

        # Initialize remaining values for each number in [low, high]
        remaining = [base + i for i in range(1, max_k + 1)]

        # Batched trial division: for each prime q <= min(p, 10^6),
        # divide out q from all multiples in [low, high].
        # This is much faster than doing trial division per number.
        limit = min(p, 10**6)
        for q in small_primes:
            if q > limit:
                break
            if q > p:
                break
            # Find first multiple of q in [low, high]
            first = ((low + q - 1) // q) * q
            if first > high:
                continue
            for n in range(first, high + 1, q):
                idx = n - low  # 0-indexed
                while remaining[idx] % q == 0:
                    remaining[idx] //= q

        # Now check each remaining value in order.
        # If remaining[i] > 1, the number has a prime factor > min(p, 10^6).
        # We need to verify whether that factor is <= p or > p.
        for i in range(1, max_k + 1):
            m = remaining[i - 1]
            if m > 1:
                if is_prime(m):
                    if m > p:
                        return i - 1  # Run breaks here
                else:
                    # m is composite; check if all its factors are <= p
                    if not is_p_smooth(m, p):
                        return i - 1
            # m == 1 means fully factored into primes <= p, so p-smooth

        return max_k

    # Generate candidate primes and test them
    best_k = 0
    best_p = 2

    # 1. Test small primes first (fast factorization, good for establishing baseline)
    for p in small_primes:
        if time.time() > deadline - 5:
            break
        k = find_smooth_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # 2. Search larger primes using segmented sieve
    def segmented_sieve(low, high):
        """Generate primes in [low, high]"""
        if high < 2:
            return []
        limit = int(math.isqrt(high)) + 1
        if limit <= 10**6 and small_primes[-1] >= limit:
            small = [p for p in small_primes if p <= limit]
        else:
            small = sieve_primes(limit)

        sieve = bytearray(b'\x01') * (high - low + 1)
        for p in small:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                sieve[j - low] = 0
        primes = []
        for i in range(low, high + 1):
            if sieve[i - low] and i >= 2:
                primes.append(i)
        return primes

    # Search ranges with adaptive step, starting from where small primes ended
    current_low = 10**6 + 1
    current_high = 10**7

    while current_high <= 10**9 and time.time() < deadline - 15:
        candidates = segmented_sieve(current_low, current_high)

        for p in candidates:
            if time.time() > deadline - 5:
                break

            # Quick pre-check: test first 3 numbers with fast method
            # to avoid expensive full run check for most candidates.
            base = p * p
            # Fast check for base+1
            m1 = base + 1
            trial_limit = min(p, 10**5)
            for q in small_primes:
                if q > trial_limit:
                    break
                while m1 % q == 0:
                    m1 //= q
            if m1 > 1 and is_prime(m1) and m1 > p:
                continue

            # Fast check for base+2
            m2 = base + 2
            for q in small_primes:
                if q > trial_limit:
                    break
                while m2 % q == 0:
                    m2 //= q
            if m2 > 1 and is_prime(m2) and m2 > p:
                continue

            # Fast check for base+3
            m3 = base + 3
            for q in small_primes:
                if q > trial_limit:
                    break
                while m3 % q == 0:
                    m3 //= q
            if m3 > 1 and is_prime(m3) and m3 > p:
                continue

            # If pre-check passes, do full check
            k = find_smooth_run(p)
            if k > best_k:
                best_k = k
                best_p = p

        # Increase range
        current_low = current_high + 1
        current_high = min(current_high * 10, 10**9)

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
