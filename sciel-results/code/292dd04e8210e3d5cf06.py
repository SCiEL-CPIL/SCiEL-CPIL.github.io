import time
import math
import random

random.seed(20240517)

# Deterministic Miller-Rabin bases valid for n < 2^64
_MR_BASES_64 = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
_SP = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def is_prime(n):
    if n < 2:
        return False
    for p in _SP:
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES_64:
        a %= n
        if a == 0:
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


def _sieve(limit):
    s = bytearray([1]) * (limit + 1)
    s[0] = s[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(2, limit + 1) if s[i]]


_SMALL = _sieve(1000)


def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n)
        c = random.randrange(1, n)
        m = 128
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
        if g != n and g > 1:
            return g


def has_large_factor(n, p):
    """Return True iff n has a prime factor strictly greater than p."""
    for sp in _SMALL:
        if sp > p:
            break
        if n % sp == 0:
            while n % sp == 0:
                n //= sp
    if n == 1:
        return False
    if n <= p:
        # every remaining prime factor is <= n <= p
        return False
    if is_prime(n):
        return True
    d = pollard_brent(n)
    if d == n:
        return True  # degenerate: treat as prime > p
    return has_large_factor(d, p) or has_large_factor(n // d, p)


def construct():
    t0 = time.monotonic()
    DEADLINE = 140.0          # safety margin well under the 150 s kill
    LIMIT = 1000000
    primes = _sieve(LIMIT)
    best_k = 0
    best_p = 7
    for p in primes:
        k = 0
        while k < 200:
            n = p * p + k + 1
            if has_large_factor(n, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
        if time.monotonic() - t0 > DEADLINE - 5:
            break
    if best_k < 1:
        best_p, best_k = 7, 1   # 49+1 = 50 = 2*5^2, all factors <= 7
    return {'p': best_p, 'k': best_k}
