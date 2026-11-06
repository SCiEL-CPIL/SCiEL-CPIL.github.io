"""Large reciprocal-subsum-free sets: greedy (multi-order) + local search with fast bitset checks."""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 88.0  # safety margin under 90s wall-clock


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _is_subsum_free(S):
    """
    Check if set S is reciprocal-subsum-free.
    For each x in S, 1/x must NOT equal sum(1/y for y in T)
    where T is a non-empty subset of S \ {x}.
    Uses integer subset-sum via bitset DP (multiply by LCM).
    """
    S_list = list(S)
    n = len(S_list)
    if n < 2:
        return True

    L = 1
    for x in S_list:
        L = _lcm(L, x)

    vals = [L // x for x in S_list]
    total = sum(vals)

    for i in range(n):
        target = vals[i]
        if target > total - target:
            continue  # impossible: others' total < target
        # bitset: bit k set => sum k achievable
        bits = 1
        for j in range(n):
            if j != i:
                bits |= bits << vals[j]
        if (bits >> target) & 1:
            return False
    return True


def _greedy(order):
    """Build a set by trying to insert elements in the given order."""
    current = set()
    for k in order:
        if _time_remaining() < 2.0:
            break
        if k in current:
            continue
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current


def _try_add(current, candidates, max_add):
    """Try to add up to max_add elements from candidates, keep best."""
    best = set(current)
    best_len = len(best)
    if not candidates or not current:
        return best
    for r in range(1, max_add + 1):
        if len(candidates) < r or _time_remaining() < 3.0:
            break
        for combo in combinations(candidates, r):
            if _time_remaining() < 2.0:
                break
            new_set = best | set(combo)
            if len(new_set) > best_len and _is_subsum_free(new_set):
                best = new_set
                best_len = len(new_set)
    return best


def _swap_improve(current, all_elements):
    """Try swaps that keep or grow cardinality: 1-for-1, 1-for-2, 2-for-3."""
    best = set(current)
    best_len = len(best)
    missing = all_elements - best
    if not missing or _time_remaining() < 5.0:
        return best
    missing = list(missing)
    cur = list(best)

    # 1-for-1 (same size, may enable later growth)
    for a in cur:
        if _time_remaining() < 2.0:
            break
        base = best - {a}
        for m in missing:
            if _time_remaining() < 1.0:
                break
            cand = base | {m}
            if _is_subsum_free(cand):
                if len(cand) > best_len:
                    best = cand
                    best_len = len(cand)

    # 1-for-2 (grow by 1)
    if len(missing) >= 2 and _time_remaining() > 8.0:
        for a in cur:
            if _time_remaining() < 3.0:
                break
            base = best - {a}
            for m1, m2 in combinations(missing, 2):
                if _time_remaining() < 1.5:
                    break
                cand = base | {m1, m2}
                if len(cand) > best_len and _is_subsum_free(cand):
                    best = cand
                    best_len = len(cand)

    # 2-for-3 (grow by 1)
    if len(missing) >= 3 and _time_remaining() > 15.0:
        for a1, a2 in combinations(cur, 2):
            if _time_remaining() < 3.0:
                break
            base = best - {a1, a2}
            for m1, m2, m3 in combinations(missing, 3):
                if _time_remaining() < 1.5:
                    break
                cand = base | {m1, m2, m3}
                if len(cand) > best_len and _is_subsum_free(cand):
                    best = cand
                    best_len = len(cand)
    return best


def _improve_once(current, all_elements):
    """One local-search pass: add then swap."""
    missing = all_elements - current
    if missing and _time_remaining() > 6.0:
        current = _try_add(current, list(missing), 4)
    if _time_remaining() > 6.0:
        current = _swap_improve(current, all_elements)
    return current


def construct():
    all_elements = set(range(1, 25))
    rng = random.Random(20240607)

    # Phase A: greedy with several orders, keep the best
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        [12, 13, 11, 14, 10, 15, 9, 16, 8, 17, 7, 18, 6, 19, 5, 20, 4, 21, 3, 22, 2, 23, 1, 24],
    ]
    for _ in range(2):
        o = list(range(1, 25))
        rng.shuffle(o)
        orders.append(o)

    best = set()
    for o in orders:
        if _time_remaining() < 5.0:
            break
        g = _greedy(o)
        if len(g) > len(best):
            best = g
    if not best:
        return [24]

    # Phase B: local search with budgeted restarts
    for _ in range(6):
        if _time_remaining() < 15.0:
            break
        improved = _improve_once(best, all_elements)
        if len(improved) > len(best):
            best = improved
        else:
            # kick with a random 1-for-1 swap to escape the local optimum
            if _time_remaining() < 10.0:
                break
            cur = list(best)
            missing = list(all_elements - best)
            if not missing:
                break
            a = rng.choice(cur)
            m = rng.choice(missing)
            cand = (best - {a}) | {m}
            if _is_subsum_free(cand):
                best = cand

    # Phase C: final polish
    if _time_remaining() > 6.0:
        best = _improve_once(best, all_elements)

    return list(best)
