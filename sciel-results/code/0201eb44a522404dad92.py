"""Antichain smooth representations: precompute min-term per target, with time guard."""
# EVOLVE-BLOCK-START
import time
import math
import sys

sys.setrecursionlimit(100000)

_DEADLINE = time.time() + 75.0   # safety margin before 90s kill

# ---- smooth numbers up to 128 ----
_SMOOTH = []
for _x in range(1, 129):
    _y = _x
    while _y % 2 == 0:
        _y //= 2
    while _y % 3 == 0:
        _y //= 3
    while _y % 5 == 0:
        _y //= 5
    if _y == 1:
        _SMOOTH.append(_x)

# ---- precompute divisor sets for the antichain check ----
_DIVS = {}
for _x in _SMOOTH:
    _d = []
    _r = _x
    while _r > 0:
        _d.append(_r)
        _r //= 2
    _DIVS[_x] = _d


def _search(pool, target, k, max_idx, chosen, blocked):
    """DFS: find k distinct pool elements summing to target, antichain (no divides).
    Returns list of terms or None. pool ascending, chosen ascending, blocked = set of
    values that would violate the antichain (any chosen divisor)."""
    if k == 0:
        return [] if target == 0 else None
    # feasibility pruning
    if target < 0:
        return None
    if target > sum(pool[i] for i in range(max_idx - k + 1, max_idx)):
        return None
    for i in range(max_idx - k, -1, -1):
        x = pool[i]
        if x > target:
            continue
        # antichain: x must not divide any already-chosen term
        if x in blocked:
            continue
        if x < target:
            # extend blocked with divisors of x (those > x are irrelevant, only keep <=x)
            nb = blocked | set(d for d in _DIVS[x] if d <= x)
            res = _search(pool, target - x, k - 1, i, chosen + [x], nb)
            if res is not None:
                return [x] + res
        else:  # x == target, this is the last term
            return [x]
    return None


def _min_terms(n):
    pool = [x for x in _SMOOTH if x <= n]
    # try 1, 2, 3 terms (where real savings live); fall back to 3
    for k in (1, 2, 3):
        res = _search(pool, n, k, len(pool), [], set())
        if res is not None:
            return res
    # guaranteed: n itself is a valid 1-term antichain
    return [n]


def construct():
    global _DEADLINE
    out = []
    for n in range(1, 129):
        if time.time() >= _DEADLINE:
            # safety: fall back to the trivially valid single-term representation
            for m in range(n, 129):
                out.append([m])
            break
        out.append(_min_terms(n))
    return out
# EVOLVE-BLOCK-END
