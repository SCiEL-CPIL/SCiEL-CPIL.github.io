import random
import time

START_TIME = time.time()
DEADLINE = 58.0  # safety margin before the 90s kill (and well under the 60s eval budget)

def _remaining():
    return DEADLINE - (time.time() - START_TIME)

def check_validity(A):
    """Fast incremental check: no subset sum is reachable at two different cardinalities.

    Returns True iff for every integer s, the set of cardinalities k for which
    s is a sum of exactly k elements of A has size at most 1.
    """
    n = len(A)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0] = {0}
    cur_max = 0
    for a in A:
        new_entries = []
        for k in range(cur_max, -1, -1):
            src = sums_by_k[k]
            if not src:
                continue
            nk = k + 1
            dst = sums_by_k[nk]
            for s in src:
                ns = s + a
                if ns in dst:
                    continue
                # conflict if ns already exists at any other cardinality
                for j in range(n + 1):
                    if j != nk and ns in sums_by_k[j]:
                        return False
                dst.add(ns)
            new_entries.append(nk)
        for nk in new_entries:
            if nk > cur_max:
                cur_max = nk
    return True

def _try_add(current, current_set, candidates):
    """Greedily add the best (smallest) candidate that keeps validity. Returns (added, new_set)."""
    for c in candidates:
        if c in current_set:
            continue
        if check_validity(current + [c]):
            current_set.add(c)
            return True, c
    return False, None

def construct():
    if _remaining() < 1.0:
        return [1, 2, 4, 8, 16, 32, 64, 128]

    # Known valid base (powers of 2 are cardinality-separated).
    base = [1, 2, 4, 8, 16, 32, 64, 128]
    best = list(base)
    best_set = set(base)

    # Primary greedy build: try to add small elements first (small values keep
    # subset sums spread out and reduce cross-cardinality collisions).
    if _remaining() > 2.0:
        current = list(base)
        current_set = set(base)
        # candidate order: small first, then large (small is the heuristic sweet spot)
        candidates = [x for x in range(1, 257) if x not in current_set]
        candidates.sort()
        while _remaining() > 1.5:
            added, c = _try_add(current, current_set, candidates)
            if not added:
                break
            current.append(c)
        if len(current) > len(best):
            best = list(current)
            best_set = set(current)

    # Second pass with the remaining candidates (larger values) to squeeze more.
    if _remaining() > 2.0:
        current = list(best)
        current_set = set(best)
        remaining = [x for x in range(1, 257) if x not in current_set]
        remaining.sort()
        while _remaining() > 1.5:
            added, c = _try_add(current, current_set, remaining)
            if not added:
                break
            current.append(c)
        if len(current) > len(best):
            best = list(current)
            best_set = set(current)

    # Random-restart phase: escape local optima by starting from a random valid
    # seed of the current best size and greedily growing.
    if _remaining() > 3.0:
        size = len(best)
        rng = random.Random(12345)
        for _ in range(200):
            if _remaining() < 1.5:
                break
            seed = sorted(rng.sample(range(1, 257), size))
            if not check_validity(seed):
                continue
            cur = list(seed)
            cur_set = set(seed)
            cands = [x for x in range(1, 257) if x not in cur_set]
            cands.sort()
            while _remaining() > 1.2:
                added, c = _try_add(cur, cur_set, cands)
                if not added:
                    break
                cur.append(c)
            if len(cur) > len(best):
                best = list(cur)
                best_set = set(cur)

    result = [int(x) for x in best]
    return result
