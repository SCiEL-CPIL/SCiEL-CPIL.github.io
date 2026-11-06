import random
import time

_N = 12
_LIMIT = 1048576

def _valid(a):
    # a must be strictly increasing positive ints with all 2^12 subset sums distinct
    n = len(a)
    if n != _N or a[0] < 1:
        return False
    for i in range(n - 1):
        if a[i] >= a[i + 1]:
            return False
    if a[-1] > _LIMIT:
        return False
    seen = {0}
    for v in a:
        new = [s + v for s in seen]
        if len(new) != len(seen):
            return False
        seen.update(new)
        if len(seen) != 2 * len(seen) - 1:
            return False
    return True

def _score(a):
    # objective = largest element (lower is better); infeasible -> large penalty
    if not _valid(a):
        return 10**18
    return a[-1]

def _greedy_seed(maxv):
    # Build a strictly increasing set greedily: pick smallest v that keeps sums distinct.
    a = []
    seen = {0}
    start = 1
    for _ in range(_N):
        for v in range(start, maxv + 1):
            new = [s + v for s in seen]
            if len(set(new)) == len(seen):
                a.append(v)
                seen.update(new)
                start = v + 1
                break
        else:
            return None
    return a

def _local_search(a, maxv, rng, deadline):
    a = list(a)
    best = _score(a)
    if best >= 10**18:
        return a
    n = _N
    # Try single-index downward moves (keep increasing, keep distinct sums)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        order = list(range(n - 1, 0, -1))
        rng.shuffle(order)
        for i in order:
            if time.time() >= deadline:
                break
            lo = a[i - 1] + 1 if i > 0 else 1
            for v in range(a[i] - 1, lo - 1, -1):
                cand = a[:i] + [v] + a[i + 1:]
                s = _score(cand)
                if s < best:
                    a = cand
                    best = s
                    improved = True
                    break
    return a

def construct():
    t0 = time.time()
    deadline = t0 + 78.0  # safety margin under the 90s wall-clock kill
    rng = random.Random(20240517)

    best = _greedy_seed(1 << 20)
    if best is None:
        return [1 << i for i in range(_N)]  # always-valid fallback (powers of two)
    best = _local_search(best, 1 << 20, rng, deadline)
    best_val = _score(best)

    # Randomized restarts with increasing max upper bounds, then polish.
    bounds = [1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19]
    for b in bounds:
        if time.time() >= deadline:
            break
        # A few restarts per bound
        for _ in range(3):
            if time.time() >= deadline:
                break
            a = _greedy_seed(b)
            if a is None:
                continue
            a = _local_search(a, b, rng, deadline)
            s = _score(a)
            if s < best_val:
                best_val = s
                best = a

    # Final high-precision polish: exhaustive single-index downward scan.
    if time.time() < deadline:
        best = _local_search(best, best_val, rng, deadline)

    return list(best)
