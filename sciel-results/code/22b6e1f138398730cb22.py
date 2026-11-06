"""
Large reciprocal-subsum-free sets — fast exact version.

Key speedups over the previous program:
  * A single global LCM L = lcm(1..24) is used for every subset-sum check,
    so the scaled integer for element x is always L//x (independent of the
    current set).  This makes the subset-sum problem depend only on (x,
    frozenset(other elements)), which we memoise globally.
  * _can_form_sum uses a bitset DP for moderate targets and meet-in-the-middle
    for very small item lists; results are cached by (target, frozenset(items)).
  * Because each subsum-freeness test is now cheap, we run many more greedy
    restarts and a deeper local search within the wall-clock budget.
"""
import time
import math
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 80.0  # safety margin under the 90 s wall-clock limit


def _time_left():
    return _DEADLINE - (time.time() - _START)


# ---- global LCM of 1..24 -------------------------------------------------
def _lcm(a, b):
    return a * b // math.gcd(a, b)


_L = 1
for _i in range(1, 25):
    _L = _lcm(_L, _i)
# _L = 5354228880
_SCALED = {x: _L // x for x in range(1, 25)}


# ---- fast subset-sum with memoisation -----------------------------------
_sum_cache = {}


def _can_form_sum(target, items):
    """True iff some non-empty subset of *items* (positive ints) sums to target."""
    if target <= 0:
        return False
    # drop items that cannot contribute
    items = tuple(sorted(x for x in items if 0 < x <= target))
    if not items:
        return False
    if sum(items) < target:
        return False
    key = (target, items)
    hit = _sum_cache.get(key)
    if hit is not None:
        return hit

    n = len(items)
    if n <= 18:
        # meet-in-the-middle
        mid = n // 2
        left, right = items[:mid], items[mid:]
        ls = {0}
        for x in left:
            ls |= {s + x for s in ls if s + x <= target}
        rs = {0}
        for x in right:
            rs |= {s + x for s in rs if s + x <= target}
        ok = any((target - s) in rs for s in ls)
    elif target <= 1_000_000:
        # bitset DP
        bits = 1
        mask = (1 << (target + 1)) - 1
        for x in items:
            bits |= (bits << x)
            bits &= mask
        ok = ((bits >> target) & 1) == 1
    else:
        # bounded set DP with early exit
        ach = {0}
        ok = False
        for x in items:
            new = {s + x for s in ach if s + x <= target}
            ach |= new
            if target in ach:
                ok = True
                break
    _sum_cache[key] = ok
    return ok


# ---- subsum-freeness with memoisation -----------------------------------
_free_cache = {}


def _is_free(S):
    """Reciprocal-subsum-free test for a set of ints in [1,24]."""
    if len(S) < 2:
        return True
    key = frozenset(S)
    hit = _free_cache.get(key)
    if hit is not None:
        return hit
    Ss = sorted(S)
    vals = [_SCALED[x] for x in Ss]
    n = len(Ss)
    ok = True
    for i in range(n):
        target = vals[i]
        others = tuple(v for j, v in enumerate(vals) if j != i)
        if _can_form_sum(target, others):
            ok = False
            break
    _free_cache[key] = ok
    return ok


# ---- constructors --------------------------------------------------------
def _greedy(order):
    cur = set()
    for k in order:
        if _time_left() < 4.0:
            break
        cand = cur | {k}
        if _is_free(cand):
            cur = cand
    return cur


def _try_add(cur, missing, max_add=2):
    best = set(cur)
    best_size = len(best)
    for r in range(1, max_add + 1):
        if _time_left() < 3.0:
            break
        for combo in combinations(missing, r):
            if _time_left() < 2.0:
                break
            ns = best | set(combo)
            if len(ns) <= best_size:
                continue
            if _is_free(ns):
                best = ns
                best_size = len(best)
    return best


def _swap_improve(cur):
    best = set(cur)
    best_size = len(best)
    missing = set(range(1, 25)) - best
    if not missing:
        return best
    for rem in list(best):
        if _time_left() < 3.0:
            break
        base = best - {rem}
        for m in missing:
            if _time_left() < 2.0:
                break
            ns = base | {m}
            if len(ns) > best_size and _is_free(ns):
                best, best_size = ns, len(ns)
        if len(missing) >= 2 and _time_left() > 6.0:
            for m1, m2 in combinations(missing, 2):
                if _time_left() < 2.0:
                    break
                ns = base | {m1, m2}
                if len(ns) > best_size and _is_free(ns):
                    best, best_size = ns, len(ns)
    return best


def _local_search(cur, passes=3):
    if not cur:
        return cur
    for _ in range(passes):
        if _time_left() < 10.0:
            break
        missing = set(range(1, 25)) - cur
        if missing:
            cur = _try_add(cur, sorted(missing, reverse=True), max_add=2)
        if _time_left() < 8.0:
            break
        cur = _swap_improve(cur)
        if not (set(range(1, 25)) - cur):
            break
    return cur


def construct():
    best = set()
    best_size = 0

    cands = []
    cands.append(_greedy(range(24, 0, -1)))
    cands.append(_greedy(range(1, 25)))
    for seed in range(40):
        if _time_left() < 6.0:
            break
        rng = random.Random(seed)
        elems = list(range(1, 25))
        rng.shuffle(elems)
        cands.append(_greedy(elems))

    for c in cands:
        if len(c) > best_size:
            best, best_size = set(c), len(c)
    if not best:
        return [24]

    imp = _local_search(best, passes=4)
    if len(imp) > best_size:
        best, best_size = imp, len(imp)

    for c in cands:
        if _time_left() < 12.0:
            break
        if len(c) >= best_size - 1:
            i2 = _local_search(c, passes=3)
            if len(i2) > best_size:
                best, best_size = i2, len(i2)

    if _time_left() > 4.0:
        for m in sorted(set(range(1, 25)) - best, reverse=True):
            ns = best | {m}
            if _is_free(ns):
                best, best_size = ns, len(ns)

    return sorted(best)
