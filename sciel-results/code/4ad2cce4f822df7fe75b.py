import math
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 113.0

    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Primes up to 10^6 for the "small factor" division
    SIEVE_LIMIT = 1000000
    primes_small = sieve(SIEVE_LIMIT)
    primes_small_np = np.array(primes_small, dtype=np.int64)

    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for sp in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == sp:
                return True
            if n % sp == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            d = 1
            c = random.randint(1, n - 1)
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        factors = []
        while n > 1:
            if n < 10000:
                # Trial division for small n
                for p in primes_small:
                    if p * p > n:
                        break
                    while n % p == 0:
                        factors.append(p)
                        n //= p
                if n > 1:
                    factors.append(n)
                break
            d = pollard_rho(n)
            if d == n:
                # Retry
                continue
            factors.extend(factorize(d))
            factors.extend(factorize(n // d))
        return factors

    def check_run(p):
        # p^2 + 1 ... p^2 + 200
        # Use int64
        p_sq = p * p
        if p_sq > np.iinfo(np.int64).max - 200:
            # Fallback to Python ints if overflow (unlikely for p <= 10^9)
            rem = [p_sq + i for i in range(1, 201)]
            for q in primes_small:
                for i in range(200):
                    while rem[i] % q == 0:
                        rem[i] //= q
            k = 0
            for i in range(200):
                r = rem[i]
                if r == 1:
                    k = i + 1
                elif is_prime_mr(r):
                    if r <= p:
                        k = i + 1
                    else:
                        break
                else:
                    facs = factorize(r)
                    if all(f <= p for f in facs):
                        k = i + 1
                    else:
                        break
            return k

        rem = np.array([p_sq + i for i in range(1, 201)], dtype=np.int64)
        
        for q in primes_small_np:
            mask = rem % q == 0
            while mask.any():
                rem[mask] //= q
                mask = rem[mask] % q == 0
        
        k = 0
        for i in range(200):
            r = int(rem[i])
            if r == 1:
                k = i + 1
            elif is_prime_mr(r):
                if r <= p:
                    k = i + 1
                else:
                    break
            else:
                facs = factorize(r)
                if all(f <= p for f in facs):
                    k = i + 1
                else:
                    break
        return k

    best_k = 0
    best_p = 2

    # Search large primes first
    # Range: 10^9 down to 10^8
    # Segment size 10^6
    SEG_SIZE = 1000000
    start = 10**9
    end = 10**8
    
    # Sieve primes up to sqrt(start) for segmented sieve
    # sqrt(10^9) ~ 31623
    limit_seg = int(math.isqrt(start)) + 1
    base_primes = sieve(limit_seg)
    
    def segmented_sieve(low, high):
        size = high - low + 1
        is_p = bytearray(b'\x01') * size
        for p in base_primes:
            start_idx = ((low + p - 1) // p) * p - low
            for j in range(start_idx, size, p):
                is_p[j] = 0
        if low == 1:
            is_p[0] = 0
        return [low + i for i in range(size) if is_p[i]]

    # Iterate segments in descending order
    high = start
    while high > end and time.time() < deadline:
        low = max(end, high - SEG_SIZE + 1)
        primes_in_seg = segmented_sieve(low, high)
        # Check in descending order
        for p in reversed(primes_in_seg):
            if time.time() > deadline:
                break
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
            # If we find a very good k, we might want to continue to see if we can do better,
            # but if k is large, it's rare.
        high = low - 1

    return {'p': best_p, 'k': best_k}
