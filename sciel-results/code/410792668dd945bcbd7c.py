"""Search for n in [1000,1000000] minimizing distinct small-prime factors in window."""
import time
import random

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23)
_BASE = 1000
_BASE_POW = [1]
for _ in range(9):
    _BASE_POW.append(_BASE_POW[-1] * _BASE)
_MAX_N = 1000000
_WIN = 24
_DEADLINE = 80.0


def _build_smooth_packed(limit):
    """packed[i] = exponent vector of the part of i made of primes < 24, base-10 packed."""
    packed = [0] * (limit + 1)
    for p in _PRIMES:
        e = 0
        for m in range(p, limit + 1, p):
            e += 1
            packed[m] += _BASE_POW[_PRIMES.index(p)] * e
            if m * p > limit:
                continue
            for m2 in range(m * p, limit + 1, m * p):
                packed[m2] += _BASE_POW[_PRIMES.index(p)] * e
    return packed


def _distinct_count(packed, lo):
    """Number of distinct nonzero packed values in packed[lo..lo+23]."""
    seen = set()
    for i in range(lo, lo + _WIN):
        v = packed[i]
        if v:
            seen.add(v)
    return len(seen)


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE

    packed = _build_smooth_packed(_MAX_N)

    best_n = None
    best_score = None

    # Pass 1: coarse scan with stride 3
    for n in range(1000, _MAX_N - _WIN + 1, 3):
        if time.time() > deadline:
            break
        s = _distinct_count(packed, n)
        if best_score is None or s < best_score:
            best_score = s
            best_n = n
            if best_score <= 1:
                break

    # Pass 2: fine scan around best if we have time
    if best_n is not None and time.time() < deadline - 3.0:
        lo = max(1000, best_n - 50)
        hi = min(_MAX_N - _WIN + 1, best_n + 50)
        for n in range(lo, hi + 1):
            if time.time() > deadline:
                break
            s = _distinct_count(packed, n)
            if s < best_score:
                best_score = s
                best_n = n
                if best_score <= 1:
                    break

    # Pass 3: randomized scan if we have time and haven't hit 1
    if best_score is not None and best_score > 1 and time.time() < deadline - 2.0:
        rng = random.Random(12345)
        for _ in range(200000):
            if time.time() > deadline:
                break
            n = rng.randint(1000, _MAX_N - _WIN)
            s = _distinct_count(packed, n)
            if s < best_score:
                best_score = s
                best_n = n
                if best_score <= 1:
                    break

    if best_n is None:
        best_n = 1000
    return best_n
