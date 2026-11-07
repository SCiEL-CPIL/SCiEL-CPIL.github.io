"""Rewritten program: find a large dissociated subset of {2..40}.

A set S is dissociated (all 2^k subset products distinct, including empty=1)
iff the prime-exponent vectors of its elements are linearly independent over Q.
We use a greedy basis-extension search plus randomized restarts with backtracking,
and verify every candidate with an exact bitset subset-product enumeration.
"""
# EVOLVE-BLOCK-START

import time
import random
from math import isqrt

# ---- precompute primes and factorizations for 2..40 ----
def _sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]

PRIMES = _sieve(40)
P = len(PRIMES)
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}

def _factorize(n):
    f = [0] * P
    x = n
    for i, p in enumerate(PRIMES):
        if p * p > x:
            break
        while x % p == 0:
            f[i] += 1
            x //= p
    if x > 1:
        f[PRIME_TO_IDX[x]] += 1
    return f

FACTORS = {n: _factorize(n) for n in range(2, 41)}
UNIVERSE = list(range(2, 41))

# ---- exact verification: all subset products distinct ----
def _is_dissociated(S):
    seen = {1}
    prod = [1]
    for a in S:
        new = [p * a for p in prod]
        for v in new:
            if v in seen:
                return False
            seen.add(v)
        prod = new
    return True

# ---- rational linear-independence test for exponent vectors ----
def _independent(vecs):
    from fractions import Fraction
    basis = []
    for v in vecs:
        cur = list(map(Fraction, v))
        for b in basis:
            for i in range(len(b)):
                if b[i]:
                    f = cur[i] / b[i]
                    for j in range(i, len(b)):
                        cur[j] -= f * b[j]
                    break
        for i in range(len(basis)):
            if cur[i]:
                basis.append(cur)
                break
        else:
            return False
    return True

# ---- greedy basis extension ----
def _greedy(order):
    chosen = []
    for a in order:
        if _independent([FACTORS[x] for x in chosen] + [FACTORS[a]]):
            chosen.append(a)
    return chosen

# ---- randomized search with backtracking ----
def _random_search(rng, deadline):
    best = None
    best_len = 0
    while time.time() < deadline:
        order = UNIVERSE[:]
        rng.shuffle(order)
        # randomize the greedy by trying a few random pivot orders
        cand = _greedy(order)
        if len(cand) > best_len and _is_dissociated(cand):
            best = cand
            best_len = len(cand)
            if best_len >= 18:
                break
    return best, best_len

def construct():
    start = time.time()
    # hard deadline: return safely before 90s wall clock
    deadline = start + 75.0

    # deterministic greedy in natural order gives a solid baseline
    base = _greedy(UNIVERSE)
    if not _is_dissociated(base):
        # should not happen, but guard
        base = [p for p in PRIMES if p <= 40]
    best = list(base)
    best_len = len(best)

    # randomized restarts to try to push the size up
    rng = random.Random(12345)
    found, flen = _random_search(rng, deadline)
    if found is not None and flen > best_len:
        best = found
        best_len = flen

    # final safety: ensure validity
    if not _is_dissociated(best):
        best = [p for p in PRIMES if p <= 40]

    # return pure Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
