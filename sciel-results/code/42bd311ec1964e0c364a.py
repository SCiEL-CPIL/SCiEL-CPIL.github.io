"""Rewritten: fast bitset incremental validity + greedy + local-search (add/remove/swap)."""
import time
import random

_T0 = time.time()
_DEADLINE = _T0 + 84.0   # safety margin under 90s wall-clock


def _fast_check(sums):
    """Return True iff all values in `sums` are distinct and no two divide each other."""
    if len(sums) != len(set(sums)):
        return False
    for a in sums:
        if a == 0:
            return False
        b = a * 2
        while b <= max(sums):
            if b in sums:
                return False
            b += a
    return True


def _is_valid(sums):
    return _fast_check(sums)


def _greedy_build(cands, rng, cur_sum, cur_sums, cur_set):
    """Greedy: add a random candidate that keeps validity. Returns (cur_set, cur_sum, cur_sums)."""
    changed = True
    while changed:
        changed = False
        cands2 = list(cands)
        rng.shuffle(cands2)
        for x in cands2:
            ns = cur_sum + x
            if ns in cur_sums:
                continue
            new_sums = list(cur_sums) + [ns]
            if _fast_check(new_sums):
                cur_set.add(x)
                cur_sum = ns
                cur_sums = new_sums
                cands.remove(x)
                changed = True
                break
    return cur_set, cur_sum, cur_sums


def _local_search(cur_set, cur_sum, cur_sums, cands, rng, iters, deadline):
    """Local search: random add / remove / swap moves. Returns best found."""
    best_set = set(cur_set)
    best_len = len(cur_set)

    for _ in range(iters):
        if time.time() > deadline:
            break
        move = rng.random()
        if move < 0.45:
            # try add
            if not cands:
                continue
            x = rng.choice(cands)
            ns = cur_sum + x
            if ns in cur_sums:
                continue
            new_sums = list(cur_sums) + [ns]
            if _fast_check(new_sums):
                cur_set.add(x)
                cur_sum = ns
                cur_sums = new_sums
                cands.remove(x)
                if len(cur_set) > best_len:
                    best_len = len(cur_set)
                    best_set = set(cur_set)
        elif move < 0.75:
            # try remove
            if len(cur_set) <= 1:
                continue
            x = rng.choice(sorted(cur_set))
            ns = cur_sum - x
            new_sums = [s for s in cur_sums if s != ns]
            cur_set.remove(x)
            cur_sum = ns
            cur_sums = new_sums
            cands.add(x)
        else:
            # try swap: remove one, add one
            if not cands or len(cur_set) <= 1:
                continue
            x = rng.choice(sorted(cur_set))
            y = rng.choice(cands)
            if x == y:
                continue
            ns = cur_sum - x + y
            new_sums = [s for s in cur_sums if s != (cur_sum - x)]
            new_sums.append(ns)
            if len(new_sums) != len(set(new_sums)):
                continue
            if _fast_check(new_sums):
                cur_set.remove(x)
                cur_set.add(y)
                cur_sum = ns
                cur_sums = new_sums
                cands.remove(y)
                cands.add(x)
                if len(cur_set) > best_len:
                    best_len = len(cur_set)
                    best_set = set(cur_set)

    return best_set


def construct():
    deadline = _DEADLINE
    rng = random.Random()
    best = [31, 32, 34]  # known valid size-3 fallback
    best_len = len(best)
    universe = list(range(1, 257))

    attempt = 0
    while time.time() < deadline:
        attempt += 1
        rng.shuffle(universe)
        cur_set = set()
        cur_sum = 0
        cur_sums = []
        cands = list(universe)

        cur_set, cur_sum, cur_sums = _greedy_build(cands, rng, cur_sum, cur_sums, cur_set)
        if len(cur_set) > best_len:
            best_len = len(cur_set)
            best = sorted(cur_set)

        # local search from current state
        ls_set = _local_search(set(cur_set), cur_sum, list(cur_sums), list(cands), rng,
                               iters=200, deadline=deadline)
        if len(ls_set) > best_len:
            best_len = len(ls_set)
            best = sorted(ls_set)

        # occasional deeper local search on the current best
        if attempt % 3 == 0 and best_len >= 5:
            cur_set = set(best)
            cur_sum = sum(best)
            cur_sums = _all_subset_sums(best)
            cands = [v for v in range(1, 257) if v not in cur_set]
            ls_set = _local_search(cur_set, cur_sum, cur_sums, cands, rng,
                                   iters=400, deadline=deadline)
            if len(ls_set) > best_len:
                best_len = len(ls_set)
                best = sorted(ls_set)

    # final safety: verify and coerce types
    best = [int(x) for x in best]
    _verify(best)
    return best


def _all_subset_sums(arr):
    arr = list(arr)
    sums = []
    m = len(arr)
    for mask in range(1, 1 << m):
        s = 0
        for i in range(m):
            if mask & (1 << i):
                s += arr[i]
        sums.append(s)
    return sums


def _verify(arr):
    """Raise if invalid (should not happen)."""
    if len(arr) != len(set(arr)):
        raise ValueError("duplicates")
    if any(not isinstance(x, int) or isinstance(x, bool) for x in arr):
        raise ValueError("type")
    if any(x < 1 or x > 256 for x in arr):
        raise ValueError("range")
    sums = _all_subset_sums(arr)
    if len(sums) != len(set(sums)):
        raise ValueError("dup sums")
    for a in sums:
        b = a * 2
        while b <= max(sums):
            if b in sums:
                raise ValueError("divisibility")
            b += a
    return True
