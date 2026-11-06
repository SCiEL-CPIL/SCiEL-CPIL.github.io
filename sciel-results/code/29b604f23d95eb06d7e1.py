# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

_DEADLINE = time.time() + 115.0
_SMALL_LIMIT = 1000000
_SMALL_PRIMES = []
_SMALL_SIEVE = None


def _sieve(n):
    b = bytearray(b'\x01') * (n + 1)
    b[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if b[i]:
            step = i
            start = i * i
            b[start::step] = b'\x00' * (((n - start) // step) + 1)
    return [i for i in range(2, n + 1) if b[i]]


_SMALL_PRIMES = _sieve(_SMALL_LIMIT)
_SMALL_SIEVE = bytearray(b'\x01') * (_SMALL_LIMIT + 1)
_SMALL_SIEVE[0:2] = b'\x00\x00'
for _p in _SMALL_PRIMES:
    if _p * _p > _SMALL_LIMIT:
        break
    _sp = _p
    _st = _p * _p
    _SMALL_SIEVE[_st::_sp] = b'\x00' * (((_SMALL_LIMIT - _st) // _sp) + 1)


_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def is_prime(n):
    n = int(n)
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        cont = False
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                cont = True
                break
        if not cont:
            return False
    return True


def _is_small_prime(n):
    return n <= _SMALL_LIMIT and _SMALL_SIEVE[n]


def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
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
        # retry


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _factor_rec(n, out):
    if n == 1:
        return
    if _is_small_prime(n):
        out.append(n)
        return
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            while n % p == 0:
                out.append(p)
                n //= p
            _factor_rec(n, out)
            return
    if is_prime(n):
        out.append(n)
        return
    d = _pollard_brent(n)
    _factor_rec(d, out)
    _factor_rec(n // d, out)


def _factorize(n):
    out = []
    _factor_rec(int(n), out)
    return out


def _is_psmooth(v, p):
    for q in _SMALL_PRIMES:
        if q > p:
            break
        if q * q > v:
            break
        if v % q == 0:
            while v % q == 0:
                v //= q
    if v > 1 and v > p:
        return False
    return True


def _run_len(p, maxk):
    k = 0
    base = p * p
    while k < maxk:
        if not _is_psmooth(base + k + 1, p):
            break
        k += 1
    return k


def _sieve_primes(a, b):
    lo = a
    hi = b
    if lo % 2 == 0:
        lo += 1
    size = (hi - lo) // 2 + 1
    res = bytearray(b'\x01') * size
    for p in _SMALL_PRIMES:
        if p * p > hi:
            break
        if p == 2:
            continue
        start = ((lo + p - 1) // p) * p
        if start % 2 == 0:
            start += p
        idx = (start - lo) // 2
        res[idx::p] = b'\x00' * (((size - 1 - idx) // p) + 1)
    out = []
    for i in range(size):
        if res[i]:
            out.append(lo + 2 * i)
    return out


def construct():
    global _DEADLINE
    random.seed(12345)
    best_p, best_k = 2, 2
    # Quick warm start on small primes.
    for p in _SMALL_PRIMES:
        if p > 20000:
            break
        if time.time() > _DEADLINE:
            break
        k = _run_len(p, 200)
        if k > best_k:
            best_p, best_k = p, k
    # Wide sweep over larger primes in increasing order.
    lo = 20001
    while lo < 10 ** 9 and time.time() < _DEADLINE:
        hi = min(lo + 200000, 10 ** 9)
        for p in _sieve_primes(lo, hi):
            if time.time() > _DEADLINE:
                break
            k = _run_len(p, 200)
            if k > best_k:
                best_p, best_k = p, k
        lo = hi + 1
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
