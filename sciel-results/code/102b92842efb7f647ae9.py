"""Few prime factors across pairwise sums: local search to minimize distinct primes."""
import math
import time
import random

# ---------- fast factorization / prime table ----------
MAXV = 20001
_spf = [0] * MAXV
for i in range(2, MAXV):
    if _spf[i] == 0:
        for j in range(i, MAXV, i):
            if _spf[j] == 0:
                _spf[j] = i

def _prime_factors(x):
    """Return set of distinct prime factors of x using SPF."""
    s = set()
    while x > 1:
        p = _spf[x]
        s.add(p)
        while x % p == 0:
            x //= p
    return s

# ---------- objective ----------
def _distinct_primes(arr):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    n = len(arr)
    union = set()
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s = ai + arr[j]
            union |= _prime_factors(s)
            if len(union) >= 100:  # early cap, we only care about small counts
                pass
    return len(union)

# ---------- candidate pools ----------
def _smooth_candidates(limit=10000):
    """Numbers of the form 2^a * 3^b * 5^c (5-smooth) up to limit, plus multiples of 6."""
    cands = set()
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                cands.add(c)
                c *= 5
            b *= 3
        a *= 2
    # also add all multiples of 6 up to limit (helps keep factor 2 and 3)
    for m in range(6, limit + 1, 6):
        cands.add(m)
    return sorted(cands)

def _initial_guess():
    """Start from 3-smooth-ish distinct values scaled to be multiples of 6."""
    # base 5-smooth numbers, ensure distinct and <=10000, prefer multiples of 6
    base = [6, 12, 24, 36, 48, 60, 96, 120, 144, 180, 240, 300]
    return list(base)

def construct():
    t0 = time.time()
    DEADLINE = 78.0  # safety margin before 90s kill

    n = 12
    best = _initial_guess()
    best_score = _distinct_primes(best)

    # Precompute candidate pool once
    pool = _smooth_candidates(10000)
    pool_set = set(pool)

    rng = random.Random(12345)

    # Local search: repeatedly try to improve by replacing one element
    improved = True
    while improved and (time.time() - t0) < DEADLINE:
        improved = False
        cur = list(best)
        cur_score = best_score
        # shuffle order of positions to diversify
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            if (time.time() - t0) > DEADLINE:
                break
            old = cur[i]
            # candidate replacements: smooth numbers + neighbors
            cands = [x for x in pool if x != old and x <= 10000]
            # add a few random multiples of 6
            for _ in range(20):
                cands.append(rng.randint(1, 10000) * 6 if rng.randint(1, 10000) * 6 <= 10000 else rng.randint(1,10000))
            rng.shuffle(cands)
            best_local = old
            best_local_score = cur_score
            for c in cands[:400]:  # cap per position for time
                if c == old:
                    continue
                if c in cur:  # must stay distinct
                    continue
                cur[i] = c
                sc = _distinct_primes(cur)
                if sc < best_local_score:
                    best_local_score = sc
                    best_local = c
            cur[i] = best_local
            if best_local != old:
                improved = True
                best = list(cur)
                best_score = best_local_score

    # Final polish: try small random restarts if time remains
    if (time.time() - t0) < DEADLINE - 5:
        for _ in range(200):
            if (time.time() - t0) > DEADLINE - 2:
                break
            cand = []
            seen = set()
            for _ in range(n):
                x = rng.choice(pool)
                if x not in seen:
                    seen.add(x)
                    cand.append(x)
            if len(cand) == n:
                sc = _distinct_primes(cand)
                if sc < best_score:
                    best = cand
                    best_score = sc

    return list(best)
