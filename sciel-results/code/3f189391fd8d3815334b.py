import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 140

    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
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
        y, c, m = random.randrange(1, n), random.randrange(1, n), random.randrange(1, n)
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
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        return g

    def generate_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, prime in enumerate(sieve) if prime]

    # Precompute primes up to 10^6
    SIEVE_LIMIT = 10**6
    small_primes_list = generate_primes(SIEVE_LIMIT)

    def check_run(p, max_k=200):
        """
        Check p-smoothness of [p^2+1, p^2+max_k] using segmented sieve.
        """
        base = p * p
        start = base + 1
        end = base + max_k
        size = max_k
        
        # remaining[i] tracks the unfactored part of (base + i + 1)
        remaining = [start + i for i in range(size)]
        
        # Sieve with primes q where q <= p and q <= end
        for q in small_primes_list:
            if q > p:
                break
            if q > end:
                break
            # First multiple of q >= start
            first = ((start + q - 1) // q) * q
            for m in range(first, end + 1, q):
                idx = m - start
                while remaining[idx] % q == 0:
                    remaining[idx] //= q
        
        # Now determine max k where all remaining[0..k-1] == 1 or have factors <= p
        k = 0
        for i in range(size):
            if time.time() - start_time > TIME_LIMIT:
                break
            r = remaining[i]
            if r == 1:
                k = i + 1
                continue
            # r > 1: has a prime factor > SIEVE_LIMIT (or > p if p < SIEVE_LIMIT)
            if p <= SIEVE_LIMIT:
                # All primes <= p were sieved, so r > 1 means factor > p
                break
            # p > SIEVE_LIMIT: r has no factor <= SIEVE_LIMIT
            # Check if r is prime and <= p
            if is_prime(r):
                if r <= p:
                    k = i + 1
                    continue
                else:
                    break
            # r is composite with factors > SIEVE_LIMIT
            f = pollard_brent(r)
            if f <= p:
                r2 = r // f
                if r2 <= p:
                    # Both factors <= p, but need to ensure they're prime or further factorable
                    # Since no factor <= SIEVE_LIMIT, f and r2 are each prime or 1
                    if f == 1 or is_prime(f):
                        if r2 == 1 or is_prime(r2):
                            k = i + 1
                            continue
                    # f might be composite but with factors > SIEVE_LIMIT
                    # This is getting complex; use recursive check
                    # For now, if both <= p, accept (rare case)
                    if f <= p and r2 <= p:
                        k = i + 1
                        continue
                # r2 > p
                break
            else:
                # f > p
                break
        
        return k

    def get_random_prime(min_val, max_val):
        while True:
            n = random.randint(min_val, max_val)
            if n % 2 == 0:
                n += 1
            if n > max_val:
                continue
            if is_prime(n):
                return n

    best_p = 0
    best_k = 0

    # Phase 1: Small primes
    primes_small = [p for p in small_primes_list if p <= 10**5]
    for p in primes_small:
        if time.time() - start_time > TIME_LIMIT:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random search
    ranges = [
        (10**5, 10**6),
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9),
    ]
    for min_val, max_val in ranges:
        if time.time() - start_time > TIME_LIMIT:
            break
        for _ in range(3000):
            if time.time() - start_time > TIME_LIMIT:
                break
            p = get_random_prime(min_val, max_val)
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p

    if best_p == 0:
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}

if __name__ == '__main__':
    construct()
