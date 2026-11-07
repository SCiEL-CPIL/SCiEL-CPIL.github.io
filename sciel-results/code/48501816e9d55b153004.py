import time
import random

def _verify(arr):
    n = len(arr)
    if n < 1 or n > 60:
        return False
    seen = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in seen:
            return False
        seen.add(v)
    # dp[k]: bitset of achievable sums with exactly k elements
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    # Check that sums for different cardinalities are disjoint
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True


def _try_add(arr, v):
    """Try adding v to arr; return True if valid."""
    n = len(arr)
    max_n = n + 1
    dp = [0] * (max_n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    # Now add v (incremental layer)
    for k in range(max_n, 0, -1):
        dp[k] |= (dp[k - 1] << v)
    # Check disjointness
    acc = 0
    for k in range(max_n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True


def _try_swap(arr, remove_idx, add_v):
    """Try replacing arr[remove_idx] with add_v; return True if the new set is valid."""
    n = len(arr)
    dp = [0] * (n + 1)
    dp[0] = 1
    for i, x in enumerate(arr):
        if i == remove_idx:
            continue
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    for k in range(n, 0, -1):
        dp[k] |= (dp[k - 1] << add_v)
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True


def _grow_greedy(start, deadline, order='ascending'):
    """Greedy growth: repeatedly add the smallest (or largest) valid element."""
    cur = list(start)
    cur_set = set(cur)
    improved = True

    while improved and time.monotonic() < deadline:
        improved = False
        if order == 'ascending':
            vals = range(1, 257)
        else:
            vals = range(256, 0, -1)
        for v in vals:
            if v in cur_set:
                continue
            if _try_add(cur, v):
                cur.append(v)
                cur_set.add(v)
                improved = True
                break
        if not improved:
            break
    return cur


def _swap_search(arr, deadline, max_rounds=20):
    """Local search: try 1-for-1 swaps to escape greedy local optima."""
    cur = list(arr)
    best = list(arr)
    best_len = len(arr)

    for rnd in range(max_rounds):
        if time.monotonic() > deadline:
            break
        cur_set = set(cur)
        swapped = False
        for i in range(len(cur)):
            if time.monotonic() > deadline:
                break
            old_v = cur[i]
            candidates = []
            for v in range(1, 257):
                if v == old_v or v in cur_set:
                    continue
                candidates.append(v)
            rng = random.Random(hash((tuple(sorted(cur)), i, rnd)) & 0xFFFFFFFF)
            rng.shuffle(candidates)
            for v in candidates[:40]:
                if time.monotonic() > deadline:
                    break
                if _try_swap(cur, i, v):
                    cur[i] = v
                    cur_set.discard(old_v)
                    cur_set.add(v)
                    swapped = True
                    grown = _grow_greedy(cur, deadline)
                    if len(grown) > best_len:
                        best = list(grown)
                        best_len = len(grown)
                    cur = list(grown)
                    break
            if swapped:
                break
        if not swapped:
            break
    return best


def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0  # safety margin under 90s

    best = None
    best_len = 0

    # Strategy 1: Try consecutive ranges from the high end
    for n in range(60, 10, -1):
        a = 257 - n
        arr = list(range(a, 257))
        if time.monotonic() > deadline - 2:
            break
        if _verify(arr):
            if best is None or len(arr) > best_len:
                best = arr
                best_len = len(arr)

    # Strategy 2: Greedy growth from best found
    if best is not None:
        g = _grow_greedy(best, deadline)
        if len(g) > best_len:
            best = g
            best_len = len(g)
        if time.monotonic() > deadline - 2:
            return best

    # Strategy 2b: Local search with swaps
    if best is not None and time.monotonic() < deadline - 5:
        s = _swap_search(best, deadline, max_rounds=15)
        if len(s) > best_len:
            best = s
            best_len = len(s)
        if time.monotonic() > deadline - 2:
            return best

    # Strategy 3: Try various structured starts
    rng = random.Random(42)

    small_starts = []
    for a1 in [1, 2, 3, 5, 8]:
        for a2 in range(a1 + 1, min(a1 + 20, 20)):
            start = [a1, a2]
            if _verify(start):
                small_starts.append(start)

    for ratio in [2, 3, 4, 5]:
        arr = [1]
        while arr[-1] * ratio <= 256:
            arr.append(arr[-1] * ratio)
        if len(arr) >= 3 and _verify(arr):
            small_starts.append(arr)

    for start in small_starts[:25]:
        if time.monotonic() > deadline - 8:
            break
        g = _grow_greedy(start, deadline)
        if len(g) > best_len:
            best = g
            best_len = len(g)
            if time.monotonic() < deadline - 5:
                s = _swap_search(best, deadline, max_rounds=8)
                if len(s) > best_len:
                    best = s
                    best_len = len(s)

    # Strategy 4: Random restarts with growth + swap
    attempts = 0
    while time.monotonic() < deadline - 3 and attempts < 400:
        attempts += 1
        n = rng.randint(15, 45)
        if rng.random() < 0.5:
            step = 256 // n
            base = list(range(1, 257, max(step, 2)))
            if len(base) >= n:
                pool = rng.sample(base, n)
            else:
                pool = rng.sample(range(1, 257), n)
        else:
            pool = rng.sample(range(1, 257), n)

        if _verify(pool):
            g = _grow_greedy(pool, deadline)
            if len(g) > best_len:
                best = g
                best_len = len(g)
                if time.monotonic() < deadline - 5:
                    s = _swap_search(best, deadline, max_rounds=8)
                    if len(s) > best_len:
                        best = s
                        best_len = len(s)
            if best is not None and best_len >= 60:
                break

    # Fallback
    if best is None:
        best = [1, 2, 4, 8, 16, 32, 64, 128]
        if not _verify(best):
            best = list(range(193, 257))

    best = [int(v) for v in best]
    if len(best) > 60:
        best = best[:60]
    return best
