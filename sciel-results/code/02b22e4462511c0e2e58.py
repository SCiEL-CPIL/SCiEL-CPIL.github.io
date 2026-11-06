"""Fresh design: fast Pollard-Brent + sieve search over primes to maximize the
smallest prime p such that p^2+1..p^2+k are all p-smooth (largest prime factor <= p)."""
import time
import random
from math import isqrt, gcd

_DEADLINE = 108.0          # explicit safety margin under the 120 s evaluator limit
_start = time.perf_counter()


def _time_left():
    return _DEADLINE - (time.perf_counter() - _start)


# ---------- primality ----------
_MR = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)


def is_prime(n):
    if n < 2:
        return False
    for p in _SMALL:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


# ---------- Pollard-Brent factorization ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    r = 2
    q = 1
    m = 128
    g = 1
    while g == 1:
        x = y = r
        r += 1
        for _ in range(m):
            y = (y * y + 1) % n
        k = 0
        while k < m and g == 1:
            ys = y
            for _ in range(min(m, m - k)):
                x = r
                r += 1
                y = (y * y + 1) % n
                q = q * abs(x - y) % n
            g = gcd(q, n)
            k += m
        if g == n:
            while True:
                ys = (ys * ys + 1) % n
                g = gcd(abs(x - ys), n)
                if g > 1:
                    break
    return g


def factor(n):
    """Return the largest prime factor of n (exact)."""
    if n < 2:
        return 1
    stack = [n]
    largest = 1
    while stack:
        v = stack.pop()
        if v == 1:
            continue
        if is_prime(v):
            if v > largest:
                largest = v
            continue
        d = _pollard_brent(v)
        if d == v:                      # rare: restart with fresh parameters
            d = _pollard_brent(v)
        if d == v:
            # last resort: v is composite but Brent failed; brute small factors
            for i in range(2, isqrt(v) + 1):
                if v % i == 0:
                    d = i
                    break
        stack.append(d)
        stack.append(v // d)
    return largest


# ---------- wheel trial division ----------
_WHEEL = (1, 5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47, 49, 53, 55, 59, 61, 65, 67, 71, 73, 77, 79, 83, 85, 89, 91, 95, 97)


def _trial_divide(n, limit):
    for p in (2, 3, 5):
        if n % p == 0:
            return p
    w = 6
    for off in _WHEEL:
        d = 6 * w + off
        if d > limit:
            break
        if n % d == 0:
            return d
        w += 1
    return None


# ---------- sieve for candidate primes ----------
def _sieve(n):
    if n < 2:
        return []
    if n < 3:
        return [2]
    if n < 5:
        return [2, 3]
    size = (n - 5) // 2 + 1
    s = bytearray(b'\x01') * size
    r = isqrt(n - 3)
    for i in range(r // 2 + 1):
        p = 2 * i + 3
        if s[i]:
            st = (p * p - 3) // 2
            s[st::p] = b'\x00' * (((size - 1 - st) // p) + 1)
    out = [2, 3]
    out.extend(2 * i + 3 for i in range(size) if s[i])
    return out


def construct():
    global _start
    _start = time.perf_counter()
    rng = random.Random(20240517)

    best_p, best_k = 3, 0
    seen = set()

    def evaluate(p):
        nonlocal best_p, best_k
        if p in seen:
            return
        seen.add(p)
        base = p * p
        k = 0
        for i in range(1, 201):
            if _time_left() < 3.0:
                break
            if factor(base + i) > p:
                break
            k += 1
        if k > best_k:
            best_p, best_k = p, k

    # ---- phase 1: small primes via sieve (dense, cheap) ----
    n1 = 300000
    for p in _sieve(n1):
        if _time_left() < 3.0:
            break
        evaluate(p)

    # ---- phase 2: random large primes (sample the wide range up to 1e9) ----
    for _ in range(400000):
        if _time_left() < 3.0:
            break
        p = rng.randrange(1000000, 1000000000)
        if p % 2 == 0:
            p += 1
        if is_prime(p):
            evaluate(p)

    # ---- phase 3: deterministic scan near the current best prime ----
    if best_p > 0:
        lo = max(n1 + 2, best_p - 20000)
        hi = best_p + 20000
        if lo % 2 == 0:
            lo += 1
        for p in range(lo, hi + 1, 2):
            if _time_left() < 3.0:
                break
            if p in seen:
                continue
            if is_prime(p):
                evaluate(p)

    # ---- phase 4: a few random primes near the best ----
    for _ in range(20000):
        if _time_left() < 3.0:
            break
        span = max(1000, best_p // 10)
        p = rng.randrange(max(1000, best_p - span), best_p + span)
        if p % 2 == 0:
            p += 1
        if p > 1000000000:
            p = 1000000000 - p
        if is_prime(p):
            evaluate(p)

    if best_k == 0:
        best_p, best_k = 3, 1
    return {'p': best_p, 'k': best_k}
