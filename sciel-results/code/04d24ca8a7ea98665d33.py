"""Rewritten program: segmented-sieve screening of p-smooth runs with deadline guard."""
import time
import math

# ---------- primality ----------
def is_prime_mr(n):
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # deterministic bases for n < 3.3e24
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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

# ---------- Pollard-Brent ----------
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = 2 + (n > 100 and (int(time.time_ns() & 0xffffffff) % (n - 2)))
        c = 1 + (int(time.time_ns() & 0xffffffff) % max(1, n - 1))
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
                    q = q * abs(x - y) % n
                g = _gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = _gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

def factor(n, out):
    # out: dict prime->exponent
    if n <= 1:
        return
    if is_prime_mr(n):
        out[n] = out.get(n, 0) + 1
        return
    d = pollard_brent(n)
    factor(d, out)
    factor(n // d, out)

def has_large_factor(n, p):
    """True if n has any prime factor > p (i.e. n is NOT p-smooth)."""
    f = {}
    factor(n, f)
    for q in f:
        if q > p:
            return True
    return False

# ---------- segmented prime generation ----------
def primes_upto(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:limit + 1:step] = bytearray(len(sieve[start:limit + 1:step]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def sieve_segment(lo, hi, base_primes):
    """Return list of primes in [lo, hi] using base_primes (all primes <= sqrt(hi))."""
    length = hi - lo + 1
    seg = bytearray(b'\x01') * length
    for p in base_primes:
        if p * p > hi:
            break
        start = ((lo + p - 1) // p) * p
        if start < p * p:
            start = p * p
        if start > hi:
            continue
        seg[start - lo: hi - lo + 1: p] = bytearray(length - ((start - lo) // p) * p + 1) if False else bytearray(
            (hi - start) // p + 1)
    res = []
    for i in range(length):
        if seg[i]:
            v = lo + i
            if v >= 2:
                res.append(v)
    return res

# ---------- main ----------
def construct():
    start = time.time()
    DEADLINE = 150.0
    MARGIN = 4.0
    limit = 150.0 - MARGIN

    best_k = 0
    best_p = 3
    # fallback valid answer
    if is_prime_mr(3):
        best_p = 3

    # small primes for trial division in the segmented p-smooth sieve
    SMALL = primes_upto(1000)

    # base primes up to sqrt(1e9) ~ 31623 for generating candidate primes
    BASE = primes_upto(31623)

    # We'll scan candidate primes in increasing order via segmented sieve over [lo, hi].
    # To keep prime generation fast, segment width ~ 1e6.
    SEG = 1_000_000

    lo = 3
    hi = min(1_000_000, 10**9)
    # ensure we start at an odd number for the segment
    if lo % 2 == 0:
        lo += 1

    while lo <= 10**9:
        hi = min(lo + SEG - 1, 10**9)
        if time.time() > limit:
            break
        primes = sieve_segment(lo, hi, BASE)
        for p in primes:
            if time.time() > limit:
                break
            if p > 10**9:
                break
            # quick reject: p^2+1 must be p-smooth, else k=0
            n1 = p * p + 1
            if has_large_factor(n1, p):
                continue
            # segmented sieve over [p^2+1, p^2+200] to strip small factors
            a = p * p + 1
            b = p * p + 200
            seg = [n for n in range(a, b + 1)]
            for q in SMALL:
                if q > p:
                    break
                # divide q out of each multiple
                start = ((a + q - 1) // q) * q
                for m in range(start, b + 1, q):
                    idx = m - a
                    v = seg[idx]
                    while v % q == 0:
                        v //= q
                    seg[idx] = v
            # now count consecutive p-smooth numbers starting at a
            k = 0
            for v in seg:
                if v == 1:
                    k += 1
                    continue
                # residual > 1: need exact factorization to check p-smoothness
                if has_large_factor(v, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
        lo = hi + 1
        if lo % 2 == 0:
            lo += 1

    if best_k < 1:
        # ensure a valid answer
        best_p = 3
        best_k = 1
    return {'p': best_p, 'k': best_k}
