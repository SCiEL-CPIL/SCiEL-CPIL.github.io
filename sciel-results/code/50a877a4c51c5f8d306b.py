"""
Sidon set construction in [1, 284] containing {1, 2, 5, 11}.

Strategy:
1. Greedy construction to get a baseline Sidon set.
2. Local search (add / remove / swap moves) with fast bitset-based
   pairwise-sum conflict checks.
3. Multiple randomized restarts within a wall-clock deadline.
4. Return the largest valid set found.

All pairwise sums a+b (a <= b) must be distinct (Sidon / B_2 property).
"""

import time
import random
import sys

# Hard deadline: evaluator kills at 150 s; use 130 s safety margin.
DEADLINE_SECONDS = 130.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_pair_sums(arr):
    """Return the set of all pairwise sums a+b with a<=b for list arr."""
    s = set()
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s.add(arr[i] + arr[j])
    return s


def _is_sidon(arr):
    """Check if arr (sorted, distinct) is a Sidon set."""
    s = _all_pair_sums(arr)
    return len(s) == len(arr) * (len(arr) + 1) // 2


def _greedy_build(prescribed, lo, hi, seed=None):
    """Greedy Sidon construction starting from prescribed elements."""
    if seed is not None:
        random.seed(seed)
    a = sorted(prescribed)
    sums = _all_pair_sums(a)
    # Try to add elements in increasing order (with some randomization)
    candidates = list(range(lo, hi + 1))
    random.shuffle(candidates)
    for x in candidates:
        if x in a:
            continue
        new_sums = {x + y for y in a} | {2 * x}
        if len(new_sums) == len(a) + 1 and not (new_sums & sums):
            a.append(x)
            sums |= new_sums
            a.sort()
    return a


def _greedy_build_ordered(prescribed, lo, hi, direction=1):
    """Greedy Sidon construction adding in ordered fashion."""
    a = sorted(prescribed)
    sums = _all_pair_sums(a)
    if direction == 1:
        candidates = range(lo, hi + 1)
    else:
        candidates = range(hi, lo - 1, -1)
    for x in candidates:
        if x in a:
            continue
        new_sums = {x + y for y in a} | {2 * x}
        if len(new_sums) == len(a) + 1 and not (new_sums & sums):
            a.append(x)
            sums |= new_sums
            a.sort()
    return a


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def _local_search(arr, lo, hi, time_limit=2.0, rng=None):
    """
    Improve a Sidon set via add, remove, and swap moves.
    Returns the best set found.
    """
    if rng is None:
        rng = random.Random()
    arr = sorted(arr)
    best = list(arr)
    best_size = len(arr)
    start = time.time()

    def check_compatible(x, current_arr, current_sums):
        """Check if x can be added to current_arr without violating Sidon."""
        new_sums = {x + y for y in current_arr} | {2 * x}
        if len(new_sums) != len(current_arr) + 1:
            return False
        if new_sums & current_sums:
            return False
        return True

    # Precompute sums
    sums = _all_pair_sums(arr)

    while time.time() - start < time_limit:
        moved = False

        # --- Try add moves ---
        # Randomly sample candidate additions
        for _ in range(50):
            x = rng.randint(lo, hi)
            if x in arr:
                continue
            if check_compatible(x, arr, sums):
                arr.append(x)
                arr.sort()
                sums = _all_pair_sums(arr)
                if len(arr) > best_size:
                    best = list(arr)
                    best_size = len(arr)
                moved = True
                break

        if moved:
            continue

        # --- Try swap moves: replace one element with another ---
        if len(arr) > 2:
            swapped = False
            for _ in range(30):
                idx = rng.randint(0, len(arr) - 1)
                old = arr[idx]
                x = rng.randint(lo, hi)
                if x == old or x in arr:
                    continue
                # Try removing old and adding x
                candidate = [v for v in arr if v != old]
                candidate.append(x)
                candidate.sort()
                if _is_sidon(candidate):
                    arr = candidate
                    sums = _all_pair_sums(arr)
                    if len(arr) > best_size:
                        best = list(arr)
                        best_size = len(arr)
                    swapped = True
                    break
            if swapped:
                continue

        # --- Try remove then re-add (kick out a bad element) ---
        if len(arr) > 4:
            improved = False
            for _ in range(20):
                idx = rng.randint(0, len(arr) - 1)
                removed = arr[idx]
                candidate = [v for v in arr if v != removed]
                # Now try to add something better
                cand_sums = _all_pair_sums(candidate)
                for x in range(lo, hi + 1):
                    if x in candidate:
                        continue
                    if check_compatible(x, candidate, cand_sums):
                        candidate.append(x)
                        candidate.sort()
                        if len(candidate) > len(arr):
                            arr = candidate
                            sums = _all_pair_sums(arr)
                            if len(arr) > best_size:
                                best = list(arr)
                                best_size = len(arr)
                            improved = True
                        break
                if improved:
                    break
            if improved:
                continue

        # Stuck: do a small random perturbation
        if len(arr) > 2:
            # Remove a random element and try to rebuild
            idx = rng.randint(0, len(arr) - 1)
            arr = [v for v in arr if v != arr[idx]]
            sums = _all_pair_sums(arr)

    return best


def _hill_climbing(arr, lo, hi, time_limit=3.0, rng=None):
    """
    More structured hill climbing: repeatedly try to add the best-fit element.
    """
    if rng is None:
        rng = random.Random()
    arr = sorted(arr)
    start = time.time()

    def can_add(x, cur, cur_sums):
        new_sums = {x + y for y in cur} | {2 * x}
        return len(new_sums) == len(cur) + 1 and not (new_sums & cur_sums)

    while time.time() - start < time_limit:
        sums = _all_pair_sums(arr)
        added = False
        # Try all candidates, pick the one that creates the most "headroom"
        best_x = None
        for x in range(lo, hi + 1):
            if x in arr:
                continue
            if can_add(x, arr, sums):
                # Heuristic: prefer x that minimizes max pairwise sum (compactness)
                if best_x is None or x < best_x:
                    best_x = x
        if best_x is not None:
            arr.append(best_x)
            arr.sort()
            added = True
        else:
            # Try swap
            swapped = False
            for i in range(len(arr)):
                old = arr[i]
                for x in range(lo, hi + 1):
                    if x == old or x in arr:
                        continue
                    candidate = [v for v in arr if v != old]
                    candidate.append(x)
                    candidate.sort()
                    if _is_sidon(candidate):
                        arr = candidate
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                break
    return arr


# ---------------------------------------------------------------------------
# Backtracking for small target sizes
# ---------------------------------------------------------------------------

def _backtrack_construct(prescribed, target_size, lo, hi, time_limit=5.0):
    """
    Try to build a Sidon set of exactly target_size containing prescribed.
    Uses backtracking with pruning.
    Returns the set if found, else None.
    """
    start = time.time()
    prescribed = sorted(prescribed)
    current = list(prescribed)
    sums = _all_pair_sums(current)

    def search(depth, max_val):
        if time.time() - start > time_limit:
            return None
        if len(current) == target_size:
            return list(current)
        remaining = target_size - len(current)
        # Need 'remaining' more elements from (max_val, hi]
        if max_val + remaining > hi:
            return None
        for x in range(max_val + 1, hi + 1):
            # Quick check
            new_sums = {x + y for y in current} | {2 * x}
            if len(new_sums) != len(current) + 1:
                continue
            if new_sums & sums:
                continue
            # Pruning: check if enough room remains
            current.append(x)
            old_sums = sums
            sums.update(new_sums)
            result = search(depth + 1, x)
            if result is not None:
                return result
            current.pop()
            sums = old_sums
        return None

    # Start after the max prescribed
    max_prescribed = max(prescribed)
    return search(0, max_prescribed)


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    """
    Return a Sidon set in [1, 284] containing {1, 2, 5, 11}, maximizing size.
    """
    start = time.time()
    prescribed = [1, 2, 5, 11]
    LO, HI = 1, 284

    best = list(prescribed)
    best_size = len(prescribed)

    def _update_best(candidate):
        nonlocal best, best_size
        if len(candidate) > best_size:
            best = list(candidate)
            best_size = len(candidate)

    # Phase 1: Greedy baselines with different strategies
    # Ordered greedy (ascending)
    g1 = _greedy_build_ordered(prescribed, LO, HI, direction=1)
    _update_best(g1)

    # Ordered greedy (descending)
    g2 = _greedy_build_ordered(prescribed, LO, HI, direction=-1)
    _update_best(g2)

    # Randomized greedy
    for seed in [42, 123, 999]:
        g = _greedy_build(prescribed, LO, HI, seed=seed)
        _update_best(g)

    # Phase 2: Local search from best greedy result
    elapsed = time.time() - start
    if elapsed < DEADLINE_SECONDS - 30:
        rng = random.Random(42)
        improved = _local_search(best, LO, HI, time_limit=min(20.0, DEADLINE_SECONDS - 35 - elapsed), rng=rng)
        _update_best(improved)

    # Phase 3: Hill climbing from multiple starting points
    elapsed = time.time() - start
    if elapsed < DEADLINE_SECONDS - 15:
        rng = random.Random(7)
        for attempt in range(3):
            base = _greedy_build(prescribed, LO, HI, seed=100 + attempt * 137)
            hc = _hill_climbing(base, LO, HI, time_limit=5.0, rng=rng)
            _update_best(hc)
            elapsed = time.time() - start
            if elapsed > DEADLINE_SECONDS - 15:
                break

    # Phase 4: Backtracking for target sizes just above current best
    elapsed = time.time() - start
    for target in range(best_size + 1, min(best_size + 4, 21)):
        if elapsed > DEADLINE_SECONDS - 10:
            break
        remaining_time = max(1.0, DEADLINE_SECONDS - 10 - elapsed)
        result = _backtrack_construct(prescribed, target, LO, HI, time_limit=remaining_time)
        if result is not None:
            _update_best(result)
            elapsed = time.time() - start

    # Phase 5: Final local search polish
    elapsed = time.time() - start
    if elapsed < DEADLINE_SECONDS - 5:
        rng = random.Random(2024)
        polish_time = max(1.0, DEADLINE_SECONDS - 6 - elapsed)
        polished = _local_search(best, LO, HI, time_limit=polish_time, rng=rng)
        _update_best(polished)

    # Ensure validity
    best = sorted(set(best))
    if not _is_sidon(best):
        # Fallback to the smallest valid set
        best = sorted(prescribed)

    return best
