import time
import math
import random
import sys
import numpy as np

def _sieve(limit):
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

_SMALL = (2,3,5,7,11,13,17,19,23,29,31,37)
_SMALL_SET = set(_SMALL)

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _SMALL:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _SMALL:
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

def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
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
    return g

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    SIEVE_LIMIT = 2000000
    primes = _sieve(SIEVE_LIMIT)
    n_primes = len(primes)

    small = np.array(primes, dtype=np.int64)
    small = small[small <= 100000]

    best_k = 0
    best_p = 2

    def run_k(base, p):
        k = 0
        for i in range(1, 201):
            n = base + i
            v = n
            rem = v % small
            if np.any(rem == 0):
                hit = small[rem == 0]
                for q in hit:
                    while v % q == 0:
                        v //= q
            if v == 1:
                k += 1
                continue
            if _is_prime_mr(v):
                if v > p:
                    break
                k += 1
                continue
            # composite leftover with all prime factors > 1e5
            ok = True
            try:
                f = _pollard_brent(v)
            except Exception:
                ok = False
            if ok:
                if f > p or (v // f) > p:
                    ok = False
            if ok:
                k += 1
            else:
                break
        return k

    # Phase 1: scan all primes up to 2e6, large to small
    for idx in range(n_primes - 1, -1, -1):
        if time.time() > deadline:
            break
        p = primes[idx]
        base = p * p
        k = run_k(base, p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: random large primes in [2e6, 1e9]
    if time.time() < deadline - 8:
        random.seed(12345)
        attempts = 0
        while time.time() < deadline - 3 and attempts < 200000:
            attempts += 1
            cand = random.randint(2000003, 10**9)
            if cand % 2 == 0:
                cand += 1
            ok = True
            for q in primes[:64]:
                if cand % q == 0:
                    ok = False
                    break
            if not ok:
                continue
            if not _is_prime_mr(cand):
                continue
            base = cand * cand
            k = run_k(base, cand)
            if k > best_k:
                best_k = k
                best_p = cand

    return {'p': best_p, 'k': best_k}
