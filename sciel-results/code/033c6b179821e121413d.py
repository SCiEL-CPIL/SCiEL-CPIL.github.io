"""New program: Precomputed small primes + optimized smoothness check with hybrid factoring."""
# EVOLVE-BLOCK-START

def construct():
    import math
    import time
    import random

    start_time = time.time()
    TIME_LIMIT = 110.0

    # ---------- Deterministic Miller-Rabin for 64-bit ----------
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
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
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # ---------- Pollard's Rho (Brent) ----------
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(1, n - 1)
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
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
            if g != n:
                return g

    # ---------- Precompute small primes up to 100000 ----------
    SIEVE_LIMIT = 100_000
    is_small_prime = [True] * (SIEVE_LIMIT + 1)
    is_small_prime[0] = is_small_prime[1] = False
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if is_small_prime[i]:
            for j in range(i * i, SIEVE_LIMIT + 1, i):
                is_small_prime[j] = False
    small_primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_small_prime[i]]

    # ---------- Optimized smoothness check ----------
    def is_p_smooth(n, p):
        """
        Returns True if all prime factors of n are <= p.
        Uses trial division by precomputed small primes, then Pollard-Brent if needed.
        """
        if n <= 1:
            return True
        if n <= p:
            return True  # All factors <= n <= p

        temp = n
        # Trial division by small primes up to min(p, 1000)
        trial_limit = min(p, 1000)
        for q in small_primes:
            if q > trial_limit:
                break
            while temp % q == 0:
                temp //= q
        if temp == 1:
            return True

        # If temp is prime, check if <= p
        if is_prime(temp):
            return temp <= p

        # Use Pollard-Brent to factor temp
        # Check if any factor > p
        def has_large_factor(x, limit):
            if x == 1:
                return False
            if x <= limit:
                return False
            if is_prime(x):
                return x > limit
            d = pollard_brent(x)
            if d == x:
                # Retry with different parameters
                return has_large_factor(x, limit)
            return has_large_factor(d, limit) or has_large_factor(x // d, limit)

        return not has_large_factor(temp, p)

    # ---------- Segmented Sieve for primes in [low, high] ----------
    def segmented_sieve(low, high):
        size = high - low + 1
        sieve = [True] * size
        limit = int(math.isqrt(high))

        for p in small_primes:
            if p > limit:
                break
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start, high + 1, p):
                sieve[j - low] = False

        primes = []
        for i in range(low, high + 1):
            if sieve[i - low]:
                primes.append(i)
        return primes

    # ---------- Search ----------
    best_k = 0
    best_p = 7  # p=7 gives k=1 as fallback

    # Phase 1: Check small primes (p < 10000)
    for p in small_primes:
        if p >= 10000:
            break
        if time.time() - start_time > TIME_LIMIT:
            break

        p2 = p * p
        k = 0
        for offset in range(1, 201):
            val = p2 + offset
            if not is_p_smooth(val, p):
                break
            k = offset

        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Check larger primes in segments
    SEG_SIZE = 50_000
    start_range = 10_000
    end_range = 5_000_000

    for low in range(start_range, end_range, SEG_SIZE):
        high = min(low + SEG_SIZE - 1, end_range)
        if time.time() - start_time > TIME_LIMIT:
            break

        primes = segmented_sieve(low, high)

        for p in primes:
            if time.time() - start_time > TIME_LIMIT:
                break

            p2 = p * p
            k = 0
            for offset in range(1, 201):
                val = p2 + offset
                if not is_p_smooth(val, p):
                    break
                k = offset

            if k > best_k:
                best_k = k
                best_p = p

    # Ensure k >= 1
    if best_k < 1:
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
