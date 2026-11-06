"""Rewritten program: pick a dense ambient set whose largest Sidon subset is small."""
import time

# EVOLVE-BLOCK-START
import itertools

_DEADLINE = 90.0
_T0 = time.time()


def _time_left():
    return _DEADLINE - 15.0 - (time.time() - _T0)  # 15s safety margin


def _candidate_sets():
    """Yield candidate 12-sets (tuples) to evaluate. Dense sets first."""
    seen = set()

    def emit(s):
        t = tuple(sorted(s))
        if len(t) == 12 and len(set(t)) == 12 and all(0 <= x <= 128 for x in t) and t not in seen:
            seen.add(t)
            yield t

    # 1) consecutive blocks starting at small offsets (densest)
    for start in range(0, 20):
        yield from emit(range(start, start + 12))
    # 2) low-value clusters
    for s in [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
    ]:
        yield from emit(s)
    # 3) a few structured / spread sets (for safety, in case dense fails)
    for s in [
        [0, 1, 4, 10, 12, 23, 25, 36, 38, 49, 51, 62],
        [0, 2, 3, 7, 12, 18, 25, 34, 44, 55, 67, 80],
        [0, 1, 3, 7, 12, 20, 30, 42, 56, 72, 90, 110],
    ]:
        yield from emit(s)


def _upper_bound(rem, used, limit):
    """Max size of a Sidon subset of `rem` given already-chosen `used`.
    Returns (bound, tight)."""
    if not rem:
        return 0, True
    if len(rem) + len(used) <= limit:
        return len(rem), False
    # count elements of rem that can still be added
    c = 0
    for x in rem:
        if all(x + y != z + w for y in used for z in used for w in used):
            c += 1
    b = c + len(used)
    if b <= limit:
        return b, False
    return len(rem) + len(used), True


def _max_sidon(arr):
    """Exact largest subset with all unordered pair sums (incl. diagonals) unique."""
    n = len(arr)
    best = [0]
    order = list(range(n))
    used = [False] * n
    memo = {}

    def rec(rem):
        if not rem:
            return
        if len(rem) + best[0] <= best[0]:
            return
        # upper bound prune
        b, tight = _upper_bound(rem, [arr[i] for i in range(n) if used[i]], best[0] + 1)
        if b <= best[0]:
            return
        key = (tuple(rem), tuple(i for i in range(n) if used[i]))
        if key in memo and memo[key] <= best[0]:
            return
        # pick the most constrained element (most collisions with used)
        def collisions(i):
            x = arr[i]
            c = 0
            for j in range(n):
                if used[j]:
                    y = arr[j]
                    for k in range(n):
                        if used[k]:
                            if x + y == arr[j] + arr[k]:
                                c += 1
            return c
        idx = min(rem, key=collisions)
        # branch: take idx
        x = arr[idx]
        ok = True
        for j in range(n):
            if used[j]:
                y = arr[j]
                for k in range(n):
                    if used[k]:
                        if x + y == arr[j] + arr[k]:
                            ok = False
                            break
                if not ok:
                    break
        new_rem = [i for i in rem if i != idx]
        if ok:
            used[idx] = True
            rec(new_rem)
            if len(new_rem) + best[0] > best[0]:
                best[0] = max(best[0], len([i for i in range(n) if used[i]]))
            used[idx] = False
        # branch: skip idx
        rec(new_rem)
        memo[key] = best[0]

    rec(order)
    return best[0]


def construct():
    best_set = None
    best_size = 10**9
    for cand in _candidate_sets():
        if _time_left() <= 0:
            break
        s = _max_sidon(cand)
        if s < best_size:
            best_size = s
            best_set = cand
        if best_size <= 4:
            break
    if best_set is None:
        return list(range(12))
    return list(best_set)


# EVOLVE-BLOCK-END
