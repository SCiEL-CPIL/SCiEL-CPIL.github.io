"""Maximize cardinality of a dissociated reciprocal subset of [1,22].

Strategy:
  Phase 1 (fast): float64 backtracking search over subsets. For each candidate
                  subset we track all reciprocal subset sums as floats and check
                  distinctness via a tolerance. This is fast and lets us explore
                  the combinatorial space to find large valid sets.
  Phase 2 (exact): verify the best float64 candidates with exact Fraction
                  arithmetic (canonical integer representation) to guarantee
                  that ALL reciprocal subset sums are truly distinct.

We keep the largest exactly-verified valid set.
"""
import time
import random
from fractions import Fraction

_DEADLINE = 78.0  # seconds; safety margin before the 90 s evaluator kill
_START = time.time()

def _deadline_hit():
    return (time.time() - _START) > _DEADLINE

def _exact_valid(subset):
    """Return True iff all reciprocal subset sums (incl. empty) are distinct.
    Uses exact integer representation: sum of 1/a over a subset = (sum of
    product-of-others) / LCM. Compare the integer numerators."""
    n = len(subset)
    if n == 0:
        return True
    from math import gcd
    # LCM of the elements
    def _lcm(a, b):
        return a // gcd(a, b) * b
    L = 1
    for a in subset:
        L = _lcm(L, a)
        # early bail if L gets astronomically large (shouldn't for [1..22])
    # For each subset, numerator = sum over a in subset of (L // a)
    # Represent subset by bitmask; there are 2^n of them.
    m = 1 << n
    seen = set()
    # precompute L//a
    w = [L // a for a in subset]
    # DP over masks: num[mask] = sum of w[i] for i in mask
    nums = [0] * m
    for mask in range(1, m):
        # take lowest set bit
        lb = mask & (-mask)
        i = lb.bit_length() - 1
        prev = mask ^ lb
        nums[mask] = nums[prev] + w[i]
    for mask in range(m):
        v = nums[mask]
        if v in seen:
            return False
        seen.add(v)
    return True

def _float_valid(subset, tol=1e-9):
    """Fast float64 check: all reciprocal subset sums distinct within tol."""
    n = len(subset)
    if n == 0:
        return True
    recip = [1.0 / a for a in subset]
    m = 1 << n
    sums = [0.0] * m
    seen = {}
    for mask in range(m):
        s = sums[mask]
        # check against existing within tol
        # use a coarse bucket to avoid O(m^2)
        key = round(s / tol)
        bucket = seen.get(key)
        if bucket is not None:
            for t in bucket:
                if abs(s - t) < tol:
                    return False
        else:
            seen[key] = []
        seen[key].append(s)
        # build children
        for i in range(n):
            if mask & (1 << i):
                continue
        # compute sums for masks with one more bit via DP instead
    # Recompute with DP for speed
    sums = [0.0] * m
    for mask in range(1, m):
        lb = mask & (-mask)
        i = lb.bit_length() - 1
        sums[mask] = sums[mask ^ lb] + recip[i]
    seen = {}
    for s in sums:
        key = round(s / tol)
        bucket = seen.get(key)
        if bucket is not None:
            for t in bucket:
                if abs(s - t) < tol:
                    return False
        else:
            seen[key] = []
        seen[key].append(s)
    return True

def _backtrack_search(full, max_k, deadline_fn, order):
    """Backtracking: try to build a valid subset of size up to max_k using
    float64 checks. order is the candidate ordering to try. Returns the largest
    valid subset found (list)."""
    best = []
    best_k = 0
    chosen = []
    sums = [0.0]  # float subset sums of chosen
    n = len(full)

    def rec(idx):
        nonlocal best, best_k
        if deadline_fn():
            return
        if len(chosen) > best_k:
            best_k = len(chosen)
            best = chosen[:]
        if best_k >= max_k:
            return
        for i in range(idx, n):
            a = order[i]
            r = 1.0 / a
            # new sums
            new_sums = [s + r for s in sums]
            # check disjoint within tol
            tol = 1e-9
            ok = True
            # quick: check new_sums against sums and among themselves
            # among themselves they're distinct if sums are distinct (translation)
            # so only need to check new_sums vs sums
            all_sums_set = {}
            for s in sums:
                key = round(s / tol)
                all_sums_set.setdefault(key, []).append(s)
            for ns in new_sums:
                key = round(ns / tol)
                bucket = all_sums_set.get(key)
                if bucket is not None:
                    for t in bucket:
                        if abs(ns - t) < tol:
                            ok = False
                            break
                if not ok:
                    break
            if not ok:
                continue
            # commit
            chosen.append(a)
            sums.extend(new_sums)
            rec(i + 1)
            chosen.pop()
            del sums[len(sums) - len(new_sums):]

    rec(0)
    return best

def construct():
    global _START
    _START = time.time()
    random.seed()
    full = list(range(1, 23))

    # Deterministic seed baseline (known valid, 9 elements).
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = seed
    best_k = len(seed)

    # Verify seed exactly (sanity)
    if _exact_valid(seed):
        pass
    else:
        best = [1]
        best_k = 1

    # Phase 1: float64 backtracking over several random orderings,
    # trying to find large valid sets.
    found_candidates = []
    max_target = 12
    while not _deadline_hit():
        order = full[:]
        random.shuffle(order)
        cand = _backtrack_search(full, max_target, _deadline_hit, order)
        if cand and len(cand) > best_k:
            found_candidates.append(cand)
            # keep the best float candidate
            if len(cand) > best_k:
                best_k = len(cand)
        # also try a greedy pass for variety
        if _deadline_hit():
            break

    # Phase 2: exact verification of the best float candidates (largest first).
    found_candidates.sort(key=lambda x: -len(x))
    for cand in found_candidates:
        if _deadline_hit():
            break
        if len(cand) <= len(best):
            break
        if _exact_valid(cand):
            best = cand
            best_k = len(cand)
            if best_k >= max_target:
                break

    # If we have spare time, try to extend the best set by adding elements
    # (local search) with exact verification.
    if not _deadline_hit():
        cur = list(best)
        improved = True
        while improved and not _deadline_hit():
            improved = False
            for a in full:
                if a in cur:
                    continue
                trial = cur + [a]
                if _exact_valid(trial):
                    cur = trial
                    improved = True
                    break
            if cur != best:
                best = cur
                best_k = len(cur)

    if best is None or len(best) == 0:
        best = [1]
    return list(best)
