"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

MAXN = 40
DEADLINE = 80.0  # safety margin well under 90s wall-clock limit


def _is_prime(a):
    if a < 2:
        return False
    r = isqrt(a)
    for d in range(2, r + 1):
        if a % d == 0:
            return False
    return True


def _factorize(a, primes):
    vec = []
    for p in primes:
        if p * p > a:
            break
        if a % p == 0:
            e = 0
            while a % p == 0:
                a //= p
                e += 1
            vec.append((p, e))
    if a > 1:
        vec.append((a, 1))
    return vec


def _build_index():
    primes = [a for a in range(2, MAXN + 1) if _is_prime(a)]
    pindex = {p: i for i, p in enumerate(primes)}
    # exponent vector as tuple indexed by primes
    vecs = {}
    for a in range(2, MAXN + 1):
        v = [0] * len(primes)
        x = a
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                e = 0
                while x % p == 0:
                    x //= p
                    e += 1
                v[pindex[p]] = e
        if x > 1:
            v[pindex[x]] = 1
        vecs[a] = tuple(v)
    return primes, vecs


def _check_and_add(vecs, a, vecs_sum):
    """Return True if adding a keeps all subset products distinct."""
    v = vecs[a]
    n = len(vecs_sum)
    for i in range(n):
        s = vecs_sum[i]
        t = tuple(s[j] + v[j] for j in range(len(v)))
        if t in vecs_sum:
            return False
    for i in range(n):
        s = vecs_sum[i]
        vecs_sum.append(tuple(s[j] + v[j] for j in range(len(v))))
    return True


def _greedy_order(candidates, vecs, rng):
    """Build a dissociated set by adding numbers in the given order."""
    vecs_sum = [tuple(0 for _ in vecs[2])]  # empty product
    chosen = []
    for a in candidates:
        if _check_and_add(vecs, a, vecs_sum):
            chosen.append(a)
    return chosen


def _random_greedy(vecs, rng):
    order = list(range(2, MAXN + 1))
    rng.shuffle(order)
    return _greedy_order(order, vecs, rng)


def _deterministic_greedy(vecs):
    # Try several heuristic orderings deterministically.
    best = []
    # ascending
    for order in (
        list(range(2, MAXN + 1)),
        list(range(MAXN, 1, -1)),
        sorted(range(2, MAXN + 1), key=lambda a: a.bit_length()),
        sorted(range(2, MAXN + 1), key=lambda a: sum(vecs[a])),
        sorted(range(2, MAXN + 1), key=lambda a: -sum(vecs[a])),
    ):
        cand = _greedy_order(order, vecs, None)
        if len(cand) > len(best):
            best = cand
    return best


def _verify(result):
    """Exact verification that all subset products are distinct."""
    if not result:
        return True
    prods = {1}
    for a in result:
        new = set()
        for p in prods:
            q = p * a
            if q in prods or q in new:
                return False
            new.add(q)
        prods |= new
    return True


def construct():
    t0 = time.time()
    primes, vecs = _build_index()

    # Deterministic baseline first (fast, gives a guaranteed valid floor).
    best = _deterministic_greedy(vecs)
    best_len = len(best)

    # Randomized greedy search to find larger dissociated sets.
    rng = random.Random(12345)
    while time.time() - t0 < DEADLINE:
        cand = _random_greedy(vecs, rng)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            if best_len >= 18:
                break  # cannot exceed 18 (max_size)

    # Final exact verification; fall back to a guaranteed-valid set if broken.
    if not _verify(best):
        # Fallback: a known-valid dissociated set (primes + some composites).
        best = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 9, 25, 49 if 49 <= 40 else 33]
        best = [a for a in best if 2 <= a <= 40]
        if not _verify(best):
            # Last resort: just the primes (always dissociated).
            best = [a for a in range(2, 41) if _is_prime(a)]

    # Enforce the 18-element cap and return a plain list of Python ints.
    result = [int(a) for a in best][:18]
    return result
# EVOLVE-BLOCK-END
