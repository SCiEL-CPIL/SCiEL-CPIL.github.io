"""Combined program: seed repair-loop baseline + inspiration set-cover/local-improvement.

Strategy:
  * Baseline (seed): start from [0, N], repair by deleting the least-populated
    residue class for each prime p <= len(tuple) whose classes are all covered.
    This is the fixed-diameter greedy deletion and is a strong baseline.
  * Refinement (inspiration): treat the problem as set-cover over the primes
    that are "full" in [0, N], then run a local-improvement pass that re-adds
    removed elements (and swaps) to push the size up, all under a wall-clock
    deadline with a safety margin.
"""
import time

N = 5000
DEADLINE_MARGIN = 30.0
T_START = time.time()


def _time_left():
    return (330.0 - DEADLINE_MARGIN) - (time.time() - T_START)


def _primes(limit):
    """Sieve of Eratosthenes: list of primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [i for i in range(2, limit + 1) if sieve[i]]


def _check_admissible(points):
    """Return True iff `points` (distinct ints) is an admissible tuple."""
    if not points:
        return True
    m = len(points)
    for p in _primes(m):
        seen = [0] * p
        for v in points:
            seen[v % p] = 1
        if all(seen):
            return False
    return True


def _greedy_repair(points):
    """Seed repair: delete least-populated class for each violating prime."""
    changed = True
    while changed:
        changed = False
        for p in _primes(len(points)):
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                points = [v for v in points if v not in smallest]
                changed = True
                break
    return points


def _local_improve(points, removed):
    """Try to re-add removed elements and perform 1-for-1 swaps to grow size."""
    best = list(points)
    best_set = set(best)
    removed_set = set(removed)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        # Re-add pass
        for x in sorted(removed_set):
            if _time_left() < 5.0:
                break
            trial = best + [x]
            if _check_admissible(trial):
                best = trial
                best_set.add(x)
                removed_set.discard(x)
                improved = True
        # Swap pass: remove one kept, add one removed
        if _time_left() < 5.0:
            break
        for x in sorted(removed_set):
            if _time_left() < 5.0:
                break
            for y in best:
                if _time_left() < 5.0:
                    break
                if y == x:
                    continue
                trial = [v for v in best if v != y] + [x]
                if _check_admissible(trial):
                    best = trial
                    best_set.discard(y)
                    best_set.add(x)
                    removed_set.discard(x)
                    removed_set.add(y)
                    improved = True
                    break
    return best


def construct():
    # ---------- Baseline: seed repair on [0, N] ----------
    points = list(range(N + 1))
    points = _greedy_repair(points)
    best = points if _check_admissible(points) else []
    removed = [x for x in range(N + 1) if x not in set(best)]

    # ---------- Refinement: local improvement ----------
    try:
        if _time_left() > 5.0:
            improved = _local_improve(best, removed)
            if _check_admissible(improved) and len(improved) > len(best):
                best = improved
    except Exception:
        pass

    # ---------- Final safety: ensure admissible, fall back to baseline ----------
    if not _check_admissible(best):
        fallback = _greedy_repair(list(range(N + 1)))
        best = fallback if _check_admissible(fallback) else []

    return list(best)
