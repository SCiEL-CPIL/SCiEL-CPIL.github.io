import time
import random

# dp[k] is a Python int bitset: bit s set iff some k-subset sums to s.
# The property "different cardinalities have disjoint sums" is exactly:
#   acc = 0; for k: if dp[k] & acc: fail; acc |= dp[k]

def _verify(arr):
    n = len(arr)
    if n < 1 or n > 60:
        return False
    seen = set()
    for v in arr:
        if type(v) is not int:
            return False
        if v < 1 or v > 256:
            return False
        if v in seen:
            return False
        seen.add(v)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True


def _try_add(arr, v):
    """Return (ok, new_dp, new_n) if adding v preserves disjointness."""
    n = len(arr)
    m = n + 1
    dp = [0] * (m + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    # add v
    for k in range(m, 0, -1):
        dp[k] |= (dp[k - 1] << v)
    acc = 0
    for k in range(m + 1):
        if dp[k] & acc:
            return False, None, None
        acc |= dp[k]
    return True, dp, m


def _grow_greedy(start, order, deadline):
    cur = list(start)
    cur_set = set(cur)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        for v in order:
            if v in cur_set:
                continue
            ok, dp, m = _try_add(cur, v)
            if ok:
                cur.append(v)
                cur_set.add(v)
                improved = True
                break
    return cur


def _reverse_greedy(deadline):
    """Start from a large target and greedily assign the largest valid ints.
    Equivalent to forward greedy but biased toward large values, which tends
    to separate the sum ranges of different cardinalities."""
    order = list(range(256, 0, -1))
    cur = []
    cur_set = set()
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        for v in order:
            if v in cur_set:
                continue
            ok, dp, m = _try_add(cur, v)
            if ok:
                cur.append(v)
                cur_set.add(v)
                improved = True
                break
    return cur


def _forward_greedy(deadline):
    order = list(range(1, 257))
    cur = []
    cur_set = set()
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        for v in order:
            if v in cur_set:
                continue
            ok, dp, m = _try_add(cur, v)
            if ok:
                cur.append(v)
                cur_set.add(v)
                improved = True
                break
    return cur


def _swap_search(arr, deadline):
    """Local search: try replacing an element with a candidate to keep validity.
    We accept a swap if it keeps the set valid; we use it to escape plateaus
    where forward greedy is stuck. Since size is fixed, we use swaps to
    reposition the set so that subsequent forward greedy can add more."""
    n = len(arr)
    if n == 0:
        return arr
    arr = list(arr)
    arr_set = set(arr)
    candidates = [x for x in range(1, 257) if x not in arr_set]
    random.shuffle(candidates)

    # Precompute current dp
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)

    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        for i in range(n):
            if time.monotonic() > deadline:
                break
            old = arr[i]
            # try candidates; pick the first that keeps validity
            for new_val in candidates[:30]:
                if new_val == old:
                    continue
                # rebuild dp without old, then add new_val
                rem = arr[:i] + arr[i+1:]
                dp2 = [0] * (n + 1)
                dp2[0] = 1
                for x in rem:
                    for k in range(n, 0, -1):
                        dp2[k] |= (dp2[k - 1] << x)
                for k in range(n, 0, -1):
                    dp2[k] |= (dp2[k - 1] << new_val)
                acc = 0
                ok = True
                for k in range(n + 1):
                    if dp2[k] & acc:
                        ok = False
                        break
                    acc |= dp2[k]
                if ok:
                    arr[i] = new_val
                    arr_set.remove(old)
                    arr_set.add(new_val)
                    candidates.remove(new_val)
                    candidates.append(old)
                    dp = dp2
                    improved = True
                    break
    return arr


def construct():
    t0 = time.monotonic()
    deadline = t0 + 82.0  # safety margin under 90s

    best = []

    # 1) Reverse greedy (bias toward large values)
    r = _reverse_greedy(deadline)
    if len(r) > len(best):
        best = r

    # 2) Forward greedy (bias toward small values)
    f = _forward_greedy(deadline)
    if len(f) > len(best):
        best = f

    # 3) Structured starts: consecutive ranges from high end
    for n in range(60, 12, -1):
        if time.monotonic() > deadline - 3:
            break
        a = 257 - n
        arr = list(range(a, 257))
        if _verify(arr) and len(arr) > len(best):
            best = arr

    # 4) Structured starts: powers-of-two-ish and spread families
    starts = []
    for ratio in [2, 3, 4]:
        arr = [1]
        while arr[-1] * ratio <= 256:
            arr.append(arr[-1] * ratio)
        starts.append(arr)
    # spread: pick evenly spaced
    for step in [4, 5, 6, 8]:
        arr = list(range(1, 257, step))
        starts.append(arr)
    # two-band
    for lo in [1, 2, 3]:
        arr = list(range(lo, lo + 32)) + list(range(256 - 31, 257))
        starts.append([x for x in arr if 1 <= x <= 256])

    for s in starts:
        if time.monotonic() > deadline - 5:
            break
        g = _grow_greedy(s, list(range(1, 257)), deadline)
        if len(g) > len(best):
            best = g

    # 5) Random restarts with growth
    rng = random.Random(12345)
    attempts = 0
    while time.monotonic() < deadline - 8 and attempts < 400:
        attempts += 1
        n = rng.randint(8, 40)
        if rng.random() < 0.5:
            step = max(2, 256 // n)
            base = list(range(1, 257, step))
            if len(base) >= n:
                pool = rng.sample(base, n)
            else:
                pool = rng.sample(range(1, 257), n)
        else:
            pool = rng.sample(range(1, 257), n)
        if _verify(pool):
            g = _grow_greedy(pool, list(range(1, 257)), deadline - 3)
            if len(g) > len(best):
                best = g
            if len(best) >= 60:
                break

    # 6) Local-search polish on the best set to escape plateaus
    if len(best) > 0 and time.monotonic() < deadline - 6:
        polished = _swap_search(best, deadline - 3)
        if len(polished) >= len(best):
            # after polishing, try to grow again
            g2 = _grow_greedy(polished, list(range(1, 257)), deadline - 1)
            if len(g2) > len(best):
                best = g2
            elif len(polished) > len(best):
                best = polished

    # Final verification and type conversion
    if best is None or len(best) == 0:
        # Conservative fallback that should be valid
        cand = list(range(193, 257))
        if _verify(cand):
            best = cand
        else:
            best = [1, 2, 4, 8, 16, 32, 64, 128]

    # Ensure Python ints
    best = [int(v) for v in best]
    # Final safety check
    if not _verify(best):
        # fallback to a guaranteed small valid set
        best = [1, 2, 4, 8, 16, 32, 64, 128]

    return best
