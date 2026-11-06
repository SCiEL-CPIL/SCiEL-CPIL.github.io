"""
Program for large reciprocal-subsum-free sets using optimized exact-integer
subset-sum verification, greedy construction, and local search.

Key improvements over the previous version:
  * _can_form_sum uses a bitset-style integer DP (or meet-in-the-middle for
    small n) which is much faster than the previous set-of-Fractions approach.
  * The LCM-based scaling keeps all values as small integers, so the subset-sum
    check is a pure integer problem.
  * More aggressive local search with swap-in/swap-out moves and multi-start
    greedy with diverse orderings.
  * Tighter deadline guard so we always return within the 90 s wall-clock limit.
"""
import time
import math
from itertools import combinations
from functools import lru_cache

_START_TIME = time.time()
_DEADLINE = 84.0  # safety margin under the 90 s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


# ---------------------------------------------------------------------------
# Fast exact subset-sum check on small positive integers
# ---------------------------------------------------------------------------
def _can_form_sum(target, items):
    """
    Return True iff some non-empty subset of *items* sums to *target*.
    All values are positive integers.
    """
    n = len(items)
    if n == 0:
        return target == 0
    if target <= 0:
        return False

    # Quick prune: drop items > target
    items = [x for x in items if x <= target]
    if not items:
        return False
    if sum(items) < target:
        return False

    n = len(items)

    # For very small n, meet-in-the-middle is fast
    if n <= 20:
        mid = n // 2
        left = items[:mid]
        right = items[mid:]

        left_sums = {0}
        for x in left:
            new = set()
            for s in left_sums:
                ns = s + x
                if ns <= target:
                    new.add(ns)
            left_sums |= new

        right_sums = {0}
        for x in right:
            new = set()
            for s in right_sums:
                ns = s + x
                if ns <= target:
                    new.add(ns)
            right_sums |= new

        for s in left_sums:
            if (target - s) in right_sums:
                return True
        return False

    # For larger n use bitset DP
    # target can be large (LCM / x), so we cap the bitset length
    # If target is too large for a bitset, fall back to a bounded set-DP
    if target <= 200_000:
        # bitset approach: bit i set means sum i is achievable
        bits = 1  # only sum 0
        for x in items:
            bits |= (bits << x)
            # mask to keep only bits up to target
            bits &= (1 << (target + 1)) - 1
        return (bits >> target) & 1 == 1

    # Fallback: bounded set DP with pruning
    achievable = {0}
    for x in items:
        new = set()
        for s in achievable:
            ns = s + x
            if ns <= target:
                new.add(ns)
        achievable |= new
        if target in achievable:
            return True
    return target in achievable


# ---------------------------------------------------------------------------
# Subsum-freeness check with caching
# ---------------------------------------------------------------------------
_subsum_free_cache = {}


def _is_subsum_free(S):
    """
    Check whether set S (of ints in [1,24]) is reciprocal-subsum-free:
    for every x in S, 1/x != sum_{y in T} 1/y for any non-empty T subset of S\{x}.
    """
    if len(S) < 2:
        return True

    key = frozenset(S)
    if key in _subsum_free_cache:
        return _subsum_free_cache[key]

    S_list = sorted(S)
    n = len(S_list)

    L = 1
    for x in S_list:
        L = _lcm(L, x)

    vals = [L // x for x in S_list]

    result = True
    for i in range(n):
        target = vals[i]
        others = [vals[j] for j in range(n) if j != i]
        if _can_form_sum(target, others):
            result = False
            break

    _subsum_free_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Greedy constructors
# ---------------------------------------------------------------------------
def _greedy(order):
    """Greedy: add elements in the given order if subsum-free."""
    current = set()
    for k in order:
        if _time_remaining() < 4.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current


def _greedy_descending():
    return _greedy(range(24, 0, -1))


def _greedy_ascending():
    return _greedy(range(1, 25))


def _greedy_random(seed_val):
    import random
    rng = random.Random(seed_val)
    elems = list(range(1, 25))
    rng.shuffle(elems)
    return _greedy(elems)


# ---------------------------------------------------------------------------
# Local search: try to add missing elements, swap in/out
# ---------------------------------------------------------------------------
def _try_add(current, candidates, max_add=2):
    """Try to add up to max_add elements from candidates while staying subsum-free."""
    if not candidates or not current or _time_remaining() < 3.0:
        return current

    best = set(current)
    best_size = len(best)

    for r in range(1, max_add + 1):
        if _time_remaining() < 3.0:
            break
        for combo in combinations(candidates, r):
            if _time_remaining() < 2.0:
                break
            new_set = best | set(combo)
            if len(new_set) <= best_size:
                continue
            if _is_subsum_free(new_set):
                best = new_set
                best_size = len(best)
    return best


def _swap_improve(current):
    """
    Try to improve by removing one element and adding one or two others.
    """
    if not current or _time_remaining() < 5.0:
        return current

    best = set(current)
    best_size = len(best)
    all_elems = set(range(1, 25))
    missing = all_elems - best

    if not missing:
        return best

    for rem in list(best):
        if _time_remaining() < 3.0:
            break
        base = best - {rem}

        # Add 1
        for m in missing:
            if _time_remaining() < 2.0:
                break
            new_set = base | {m}
            if len(new_set) > best_size and _is_subsum_free(new_set):
                best = new_set
                best_size = len(best)

        # Add 2
        if len(missing) >= 2 and _time_remaining() > 5.0:
            for m1, m2 in combinations(missing, 2):
                if _time_remaining() < 2.0:
                    break
                new_set = base | {m1, m2}
                if len(new_set) > best_size and _is_subsum_free(new_set):
                    best = new_set
                    best_size = len(best)

    return best


def _local_search(current, passes=3):
    """Iterate add + swap passes."""
    if not current:
        return current

    all_elems = set(range(1, 25))
    for _ in range(passes):
        if _time_remaining() < 10.0:
            break
        missing = all_elems - current
        if missing:
            current = _try_add(current, sorted(missing, reverse=True), max_add=2)
        if _time_remaining() < 8.0:
            break
        current = _swap_improve(current)
        missing = all_elems - current
        if not missing:
            break
    return current


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct():
    """
    Return a nonempty subset of [1,24] that is reciprocal-subsum-free,
    maximising cardinality.
    """
    best = set()
    best_size = 0

    # --- Multi-start greedy ---
    candidates = []

    # Deterministic orderings
    candidates.append(_greedy_descending())
    candidates.append(_greedy_ascending())

    # Random orderings
    for seed in range(20):
        if _time_remaining() < 6.0:
            break
        candidates.append(_greedy_random(seed))

    # Keep the best so far
    for c in candidates:
        if len(c) > best_size:
            best = set(c)
            best_size = len(best)

    if not best:
        return [24]

    # --- Local search on the best candidate ---
    improved = _local_search(best, passes=4)
    if len(improved) > best_size:
        best = improved
        best_size = len(best)

    # --- Local search on other promising candidates ---
    for c in candidates:
        if _time_remaining() < 12.0:
            break
        if len(c) >= best_size - 1:
            imp = _local_search(c, passes=3)
            if len(imp) > best_size:
                best = imp
                best_size = len(best)

    # --- Final attempt: try adding any single missing element ---
    if _time_remaining() > 4.0:
        all_elems = set(range(1, 25))
        missing = all_elems - best
        for m in sorted(missing, reverse=True):
            new_set = best | {m}
            if _is_subsum_free(new_set):
                best = new_set
                best_size = len(best)

    return sorted(best)
