import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d, s = n - 1, 0
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

    # Pollard-Brent
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
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
                        q = q * abs(x - y) % n
                    g = isqrt(q) if q > 0 else 1 # isqrt of 0 is 0, but q=0 means gcd(n,0)=n
                    # Actually standard brent uses gcd(q, n)
                    g = 1
                    if q != 0:
                        # Compute gcd
                        a, b = q, n
                        while b:
                            a, b = b, a % b
                        g = a
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = 1
                    a, b = abs(x - ys), n
                    while b:
                        a, b = b, a % b
                    g = a
            if g != n:
                return g
            # If g == n, retry
            continue

    def factor(n, res):
        if n == 1:
            return
        if is_prime(n):
            res.append(n)
            return
        d = pollard_brent(n)
        factor(d, res)
        factor(n // d, res)

    def is_p_smooth(n, p):
        if n == 1:
            return True
        if n > p and is_prime(n):
            return False
        factors = []
        factor(n, factors)
        for f in factors:
            if f > p:
                return False
        return True

    # Precompute small primes for trial division to speed up factorization
    SIEVE_LIMIT = 10**6
    sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(SIEVE_LIMIT) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, SIEVE_LIMIT + 1) if sieve[i]]

    def is_p_smooth_optimized(n, p):
        if n == 1:
            return True
        # Trial division by small primes
        temp_n = n
        for q in small_primes:
            if q > p:
                break
            if q * q > temp_n:
                break
            while temp_n % q == 0:
                temp_n //= q
        if temp_n == 1:
            return True
        if temp_n > p:
            # Check if temp_n is prime
            if is_prime(temp_n):
                return False
            # If composite, it might have factors <= p?
            # Wait, we divided by all primes <= min(p, sqrt(temp_n_original? no))
            # We divided by primes up to min(p, sqrt(temp_n_current))
            # If temp_n > 1, it has a prime factor > last_q.
            # If last_q < p, it's possible the factor is <= p.
            # So we must factorize temp_n completely.
            factors = []
            factor(temp_n, factors)
            for f in factors:
                if f > p:
                    return False
            return True
        else:
            # temp_n <= p. Since we divided by all primes <= p that divide n?
            # No, we stopped if q*q > temp_n.
            # If temp_n <= p, and it's not 1, it must be prime (since no small factors left).
            # And prime <= p is allowed.
            return True

    best_k = 0
    best_p = 2

    # Generate primes up to 10^7
    # 10^7 is ~100MB for bytearray, might be too much? 10^7 is 10MB. OK.
    # But we can just use the small_primes list up to 10^6 and then check primality for larger?
    # Or generate more.
    # Let's generate up to 5*10^6 to be safe on memory/time.
    GEN_LIMIT = 5 * 10**6
    if GEN_LIMIT > SIEVE_LIMIT:
         sieve2 = bytearray(b'\x01') * (GEN_LIMIT + 1)
         sieve2[0] = sieve2[1] = 0
         for i in range(2, isqrt(GEN_LIMIT) + 1):
             if sieve2[i]:
                 sieve2[i*i::i] = bytearray(len(sieve2[i*i::i]))
         primes = [i for i in range(2, GEN_LIMIT + 1) if sieve2[i]]
    else:
         primes = small_primes

    for p in primes:
        if time.time() > DEADLINE:
            break
        
        # Check p^2 + 1
        N = p * p + 1
        if is_p_smooth_optimized(N, p):
            k = 1
            # Check further
            for j in range(2, 201):
                if time.time() > DEADLINE:
                    break
                N = p * p + j
                if is_p_smooth_optimized(N, p):
                    k = j
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p

    # If we didn't find anything, return a valid small one
    if best_k == 0:
        # Find a small p with k>=1
        # p=7 gives k=1?
        # 7^2+1 = 50 = 2*5*5. Max factor 5 <= 7. OK.
        # 7^2+2 = 51 = 3*17. 17 > 7. Fail.
        # So p=7, k=1 is valid.
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
