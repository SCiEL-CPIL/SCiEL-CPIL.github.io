# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 115.0
    best_k = 0
    best_p = 2

    # ---------- deterministic Miller–Rabin for 64-bit ----------
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        for w in witnesses:
            if n == w:
                return True
            if n % w == 0:
                return False
        d, r = n - 1, 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in witnesses:
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

    # ---------- Pollard Brent ----------
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y, c, m = 2, 1, 128
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
            c += 1

    # ---------- is n p-smooth (all prime factors <= p) ----------
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        if n <= p:
            return True
        if n <= p * p:
            return False
        d = 2
        while d * d <= n and d <= p:
            while n % d == 0:
                n //= d
            d += 1
        if n == 1:
            return True
        if n <= p:
            return True
        if n <= p * p:
            return False
        if is_prime(n):
            return n <= p
        a = brent(n)
        stack = [a, n // a]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if m <= p:
                continue
            if m <= p * p:
                return False
            if is_prime(m):
                return False
            b = brent(m)
            stack.append(b)
            stack.append(m // b)
        return True

    # ---------- sieve primes ----------
    def primes_up_to(limit):
        if limit < 2:
            return []
        sieve = bytearray([1]) * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
        return [i for i in range(2, limit + 1) if sieve[i]]

    # ---------- Wheel sieve: find bad offsets for a given p ----------
    # n = p^2 + i is NOT p-smooth iff it has a prime factor q > p.
    # Any such q satisfies q <= sqrt(p^2 + 200). For p >= 2, sqrt(p^2+200) < p + 100.
    # So q is in (p, p + 100]. Very few primes in that range for large p.
    def find_bad_offsets(p):
        bad = bytearray(201)  # bad[i] = 1 means p^2+i has a prime factor > p
        upper = int(math.isqrt(p * p + 200)) + 1
        for q in range(p + 1, upper + 1):
            if q % 2 == 0:
                continue
            if not is_prime(q):
                continue
            r = (-(p * p) % q) % q
            if r == 0:
                i_start = q
            else:
                i_start = r
            if i_start > 200:
                continue
            for i in range(i_start, 201, q):
                bad[i] = 1
        return bad

    # ---------- Phase 1: dense search small primes (no wheel, direct factorization) ----------
    p1 = primes_up_to(100000)
    for p in p1:
        if time.time() > deadline:
            break
        k = 0
        p2_val = p * p
        while k < 200 and is_p_smooth(p2_val + 1 + k, p):
            k += 1
        if k > best_k:
            best_k, best_p = k, p

    # ---------- Phase 2: dense search medium/large primes with wheel pre-filter ----------
    p2 = primes_up_to(50000000)
    start = len(p1)
    for i in range(start, len(p2)):
        if time.time() > deadline:
            break
        p = p2[i]
        bad = find_bad_offsets(p)
        # Find longest prefix of non-bad offsets
        k = 0
        while k < 200 and not bad[k + 1]:
            k += 1
        if k == 0:
            continue
        # Verify with full factorization
        p2_val = p * p
        verified_k = 0
        while verified_k < k and is_p_smooth(p2_val + 1 + verified_k, p):
            verified_k += 1
        if verified_k > best_k:
            best_k, best_p = verified_k, p

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
