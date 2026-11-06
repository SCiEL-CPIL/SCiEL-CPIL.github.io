"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = time.time() + 85.0  # safety margin under the 90s limit


def _is_sidon_with_diagonals(subset):
    """Check if a subset has all unordered pair sums (including diagonals) unique."""
    sums = set()
    k = len(subset)
    for i in range(k):
        for j in range(i, k):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _largest_sidon(arr):
    """Exact largest subset with all unordered pair sums (incl. diagonals) unique.
    Checks combinations from largest size down, stops at first valid one."""
    n = len(arr)
    # Try from largest possible size down to 1
    for k in range(n, 0, -1):
        for comb in combinations(arr, k):
            if _is_sidon_with_diagonals(comb):
                return k
    return 0


def _candidate(arr):
    return _largest_sidon(arr)


def _local_search(arr, rng, budget):
    arr = list(arr)
    cur = _candidate(arr)
    best = cur
    best_arr = list(arr)
    t = 1.0
    tmin = 0.02
    n = len(arr)
    steps = 0
    while time.time() < budget:
        steps += 1
        # Choose mutation type: 60% single replace, 40% swap
        if rng.random() < 0.6:
            i = rng.randrange(n)
            old = arr[i]
            arr[i] = rng.randrange(0, 129)
            if arr[i] in arr[:i] or arr[i] in arr[i + 1:]:
                arr[i] = old
                continue
            new = _candidate(arr)
            if new <= cur or rng.random() < pow(2.718281828459045, -(new - cur) / t):
                cur = new
                if new < best:
                    best = new
                    best_arr = list(arr)
            else:
                arr[i] = old
        else:
            # Swap two elements with random values
            i, j = rng.sample(range(n), 2)
            old_i, old_j = arr[i], arr[j]
            arr[i] = rng.randrange(0, 129)
            arr[j] = rng.randrange(0, 129)
            if len(set(arr)) < n:
                arr[i], arr[j] = old_i, old_j
                continue
            new = _candidate(arr)
            if new <= cur or rng.random() < pow(2.718281828459045, -(new - cur) / t):
                cur = new
                if new < best:
                    best = new
                    best_arr = list(arr)
            else:
                arr[i], arr[j] = old_i, old_j
        t = max(tmin, t * 0.998)
    return best, best_arr


def _seed_candidates(rng):
    cands = []
    # dense low cluster (maximizes sum collisions)
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    cands.append(list(base))
    # Very dense: 0..11 but shifted
    cands.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    cands.append([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    # Dense cluster + one far anchor
    for anchor in [64, 80, 96, 112, 120, 128]:
        cands.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, anchor])
    # Two dense clusters
    cands.append([0, 1, 2, 3, 4, 5, 64, 65, 66, 67, 68, 69])
    cands.append([0, 1, 2, 3, 4, 5, 32, 33, 34, 35, 36, 37])
    # Arithmetic progressions (create many a+b=c+d)
    for step in (2, 3, 4, 5, 6, 7, 8):
        cands.append([i * step for i in range(12) if i * step <= 128][:12])
    # Near-duplicates: pairs that are very close
    cands.append([0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21])
    cands.append([0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35])
    # Random
    for _ in range(30):
        cands.append(rng.sample(range(0, 129), 12))
    # Structured: multiples + jitter
    for step in (5, 7, 9):
        cands.append([(i * step) % 128 for i in range(12)])
    return cands


def construct():
    rng = random.Random(20240517)
    best = 12
    best_arr = list(range(12))
    budget = _DEADLINE - 1.0   # leave margin for return
    
    # Phase 1: Evaluate seed candidates
    for cand in _seed_candidates(rng):
        if time.time() > budget:
            break
        v = _candidate(cand)
        if v < best:
            best = v
            best_arr = list(cand)
    
    # Phase 2: Local search refinement with restarts
    while time.time() < budget - 5.0:
        # Run local search from current best
        v, va = _local_search(best_arr, rng, min(budget - 2.0, time.time() + 10.0))
        if v < best:
            best = v
            best_arr = va
        if best <= 5:
            break
        # Occasionally restart from a new random point
        if rng.random() < 0.15:
            new_start = rng.sample(range(0, 129), 12)
            v2 = _candidate(new_start)
            if v2 < best:
                best = v2
                best_arr = list(new_start)
    
    # Final polish: short local search from best
    if time.time() < budget - 1.0:
        v, va = _local_search(best_arr, rng, budget - 0.5)
        if v < best:
            best = v
            best_arr = va
    
    return sorted(best_arr)
# EVOLVE-BLOCK-END
