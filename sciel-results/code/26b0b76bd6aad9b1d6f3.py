import math
import time
import sys
import numpy as np

# ---------- prime generation (segmented sieve) ----------
def _sieve_small(limit):
    limit = int(limit)
    if limit < 2:
        return np.array([], dtype=np.int64)
    s = np.ones(limit + 1, dtype=bool)
    s[0:2] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0].astype(np.int64)

def _seg_sieve(lo, hi, small):
    lo = int(lo); hi = int(hi)
    if hi < lo:
        return np.array([], dtype=np.int64)
    size = hi - lo + 1
    s = np.ones(size, dtype=bool)
    for p in small:
        p = int(p)
        if p * p > hi:
            break
        start = ((lo + p - 1) // p) * p
        if start == p:
            start += p
        if start <= hi:
            s[start - lo::p] = False
    idx = np.nonzero(s)[0]
    return (lo + idx).astype(np.int64)

# ---------- deterministic Miller-Rabin ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def _is_prime(n):
    n = int(n)
    if n < 2:
        return False
    for p in _MR_BASES:
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

# ---------- core search ----------
def construct():
    t0 = time.time()
    DEADLINE = 118.0
    SIEVE_LIMIT = 10 ** 9
    SMALL_LIMIT = 100000

    def time_ok():
        return (time.time() - t0) < DEADLINE

    # primes up to 1e5 for trial division of candidates
    small = _sieve_small(SMALL_LIMIT)
    small_list = small.tolist()

    def is_prime_fast(n):
        n = int(n)
        if n < 2:
            return False
        for p in small_list:
            if p * p > n:
                return True
            if n % p == 0:
                return n == p
        return _is_prime(n)

    # primes up to 1e9 via segmented sieve
    seg = 10 ** 7
    prime_chunks = []
    lo = 2
    hi = min(seg, SIEVE_LIMIT)
    while lo <= SIEVE_LIMIT:
        if not time_ok():
            break
        chunk = _seg_sieve(lo, hi, small)
        if len(chunk):
            prime_chunks.append(chunk)
        lo = hi + 1
        hi = min(lo + seg, SIEVE_LIMIT)
    if not prime_chunks:
        return {'p': 3, 'k': 1}

    # search from largest primes downward (long runs more likely for large p)
    best_k = 0
    best_p = 3
    nchunks = len(prime_chunks)
    for ci in range(nchunks - 1, -1, -1):
        if not time_ok():
            break
        chunk = prime_chunks[ci]
        # process chunk from largest to smallest prime
        for j in range(len(chunk) - 1, -1, -1):
            p = int(chunk[j])
            if p > SIEVE_LIMIT:
                continue
            if not is_prime_fast(p):
                continue
            p2 = p * p
            k = 0
            while k < 200:
                N = p2 + k + 1
                if not time_ok():
                    break
                # trial divide by small primes (skip 2: N odd)
                m = N
                for q in small_list:
                    if q == 2:
                        continue
                    if q > p:
                        break
                    if q * q > m:
                        break
                    if m % q == 0:
                        m //= q
                        while m % q == 0:
                            m //= q
                # remaining m: 1 (smooth) or prime (fail) or composite (fail, since all factors > p)
                if m != 1 and (m > p or _is_prime(m)):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
    return {'p': best_p, 'k': best_k}
