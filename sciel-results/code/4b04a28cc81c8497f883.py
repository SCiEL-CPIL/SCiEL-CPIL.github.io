"""Rewritten program: fast factorization + wide prime search with deadline guard."""
import math
import time

_DEADLINE = time.time() + 135  # 150s kill; keep 15s margin


# ---------- deterministic Miller-Rabin for n < 2^64 ----------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def _is_prime(n):
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
    for a in _MR_BASES:
        if a % n == 0:
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


# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = 2
        r = 1
        q = 1
        m = 128
        g = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + 1) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + 1) % n
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + 1) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g


def _factorize(n):
    """Return list of prime factors (with multiplicity) of n."""
    factors = []
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if _is_prime(x):
            factors.append(x)
            continue
        d = _pollard_brent(x)
        stack.append(d)
        stack.append(x // d)
    return factors


# ---------- sieve for small primes ----------
def _sieve(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


_SMALL_PRIMES = _sieve(1000)


def _smallest_factor(v, primes):
    """Return the smallest prime factor of v, or v if v is prime.
    Uses trial division by primes, then Miller-Rabin, then Pollard-Brent."""
    for p in primes:
        if p * p > v:
            break
        if v % p == 0:
            return p
    if _is_prime(v):
        return v
    return _pollard_brent(v)


def _largest_factor(v, primes):
    """Return the largest prime factor of v."""
    m = v
    while m > 1:
        f = _smallest_factor(m, primes)
        if f > m:  # shouldn't happen, but safety
            f = m
        m //= f
    return m


def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill

    # Sieve primes up to 50000 for fast trial division
    primes = _sieve(50000)

    best_k = 0
    best_p = 2

    # Search primes p in increasing order
    # For each p, compute run length k of p^2+1, ..., p^2+k all p-smooth
    # Skip p if time is tight
    for p in primes:
        if time.time() > deadline - 5:
            break
        if p > 10**9:
            break
        p2 = p * p
        k = 0
        while k < 200:
            if time.time() > deadline - 2:
                break
            v = p2 + k + 1
            # Check if all prime factors of v are <= p
            m = v
            ok = True
            while m > 1:
                f = _smallest_factor(m, primes)
                if f > p:
                    ok = False
                    break
                m //= f
            if not ok:
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            # Early stop: k can't exceed 200
            if best_k >= 200:
                break

    return {'p': best_p, 'k': best_k}
