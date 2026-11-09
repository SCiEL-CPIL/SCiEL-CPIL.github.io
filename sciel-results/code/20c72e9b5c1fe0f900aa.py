# EVOLVE-BLOCK-START
import time
from math import isqrt

_START = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

def _prime_factors(n):
    """Return set of prime factors of n (n>=2)."""
    factors = set()
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            factors.add(f)
            while n % f == 0:
                n //= f
        f += 2
    if n > 1:
        factors.add(n)
    return factors

def _sum_prime_union(arr):
    """Union of prime factors over all pairwise sums of distinct elements."""
    s = set()
    m = len(arr)
    for i in range(m):
        ai = arr[i]
        for j in range(i + 1, m):
            s |= _prime_factors(ai + arr[j])
    return s

def _count(arr):
    return len(_sum_prime_union(arr))

def construct():
    best = list(range(1, 13))
    best_c = _count(best)

    # Candidate pool: powers of 2 up to 10000, plus a small set of helpful extras
    pows = [1 << k for k in range(0, 14) if (1 << k) <= 10000]  # 1..8192
    # Exhaustive over 12-subsets of powers of 2 (C(14,12)=91)
    from itertools import combinations
    for combo in combinations(pows, 12):
        c = _count(list(combo))
        if c < best_c:
            best_c = c
            best = list(combo)
        if time.time() > _DEADLINE:
            return best

    # Local search: try swapping one element for a candidate to reduce prime union
    # Candidate replacement pool: powers of 2 plus numbers of form 2^a * p where p in small prime set
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    cand_pool = set(pows)
    for p in small_primes:
        v = 1
        while v * p <= 10000:
            cand_pool.add(v * p)
            v <<= 1
    cand_pool.discard(0)
    cand_pool = sorted([x for x in cand_pool if 1 <= x <= 10000])

    cur = list(best)
    cur_c = best_c
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        cur_set = set(cur)
        for idx in range(12):
            old = cur[idx]
            for repl in cand_pool:
                if repl == old or repl in cur_set:
                    continue
                trial = cur[:idx] + [repl] + cur[idx + 1:]
                c = _count(trial)
                if c < cur_c:
                    cur = trial
                    cur_c = c
                    improved = True
                    cur_set = set(cur)
                    break
            if time.time() > _DEADLINE:
                break
        if cur_c < best_c:
            best_c = cur_c
            best = list(cur)

    return best
# EVOLVE-BLOCK-END
