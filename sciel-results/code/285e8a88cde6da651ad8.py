import time
import random


class Verifier:
    __slots__ = ("n", "dp", "acc", "valid", "max_k")

    def __init__(self, n):
        self.n = n
        self.dp = [0] * (n + 1)
        self.dp[0] = 1
        self.acc = 1
        self.valid = True
        self.max_k = 0

    def add(self, x):
        if not self.valid:
            return False
        n = self.n
        prev = self.dp
        ndp = list(prev)
        new_max_k = min(self.max_k + 1, n)
        for k in range(new_max_k, 0, -1):
            ndp[k] = prev[k] | (prev[k - 1] << x)
        for k in range(1, new_max_k + 1):
            new_sums = ndp[k] & ~prev[k]
            if new_sums & self.acc:
                self.valid = False
                return False
        for k in range(1, new_max_k + 1):
            self.acc |= ndp[k]
        self.dp = ndp
        self.max_k = new_max_k
        return True


def _check(arr):
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


def _try_add(arr, used, ver, v):
    if v in used:
        return False
    arr.append(v)
    used.add(v)
    ok = ver.add(v)
    if not ok:
        arr.pop()
        used.discard(v)
    return ok


def _greedy(deadline, rng, order=None):
    cur = []
    used = set()
    ver = Verifier(60)
    if order is None:
        order = list(range(256, 0, -1))
        if rng is not None:
            rng.shuffle(order)
    added = True
    while added and len(cur) < 60 and time.monotonic() < deadline:
        added = False
        for v in order:
            if _try_add(cur, used, ver, v):
                added = True
                break
    extra = list(range(1, 257))
    if rng is not None:
        rng.shuffle(extra)
    for v in extra:
        if time.monotonic() > deadline:
            break
        if _try_add(cur, used, ver, v):
            pass
    return cur


def _local_search(arr, deadline):
    if len(arr) >= 60:
        return arr
    cur = list(arr)
    used = set(cur)
    ver = Verifier(60)
    for v in cur:
        ver.add(v)

    # Pass A: 1-for-2
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for i in range(n):
            if time.monotonic() > deadline:
                break
            removed = cur[i]
            cur.pop(i)
            used.discard(removed)
            ver2 = Verifier(60)
            for v in cur:
                ver2.add(v)
            added_count = 0
            for v in range(256, 0, -1):
                if v not in used and _try_add(cur, used, ver2, v):
                    added_count += 1
                    if added_count == 2:
                        break
                if time.monotonic() > deadline:
                    break
            if added_count >= 2:
                improved = True
            else:
                for _ in range(added_count):
                    cur.pop()
                cur.insert(i, removed)
                used.add(removed)

    # Pass B: 1-for-1 swap to unblock
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for i in range(n):
            if time.monotonic() > deadline:
                break
            old = cur[i]
            for new in range(256, 0, -1):
                if new == old or new in used:
                    continue
                cur[i] = new
                used.discard(old)
                used.add(new)
                ver2 = Verifier(60)
                ok = True
                for v in cur:
                    if not ver2.add(v):
                        ok = False
                        break
                if not ok:
                    cur[i] = old
                    used.discard(new)
                    used.add(old)
                    continue
                added = False
                for v in range(256, 0, -1):
                    if v not in used and _try_add(cur, used, ver2, v):
                        added = True
                        break
                if added:
                    improved = True
                    break
                else:
                    cur[i] = old
                    used.discard(new)
                    used.add(old)
            if improved:
                break

    # Pass C: remove-then-re-add polish (replace element with smaller to free sum-space)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for i in range(n):
            if time.monotonic() > deadline:
                break
            old = cur[i]
            cur.pop(i)
            used.discard(old)
            ver2 = Verifier(60)
            for v in cur:
                ver2.add(v)
            found = False
            for new in range(1, 257):  # try smaller values first to free high sum-space
                if new == old or new in used:
                    continue
                if _try_add(cur, used, ver2, new):
                    found = True
                    break
            if found:
                improved = True
                break
            else:
                cur.insert(i, old)
                used.add(old)

    # Final: try to add more after all passes
    for v in range(256, 0, -1):
        if time.monotonic() > deadline:
            break
        if _try_add(cur, used, ver, v):
            pass

    return cur


def construct():
    t0 = time.monotonic()
    deadline = t0 + 82.0
    best = None
    best_len = 0

    def consider(arr):
        nonlocal best, best_len
        if _check(arr) and len(arr) > best_len:
            best = list(arr)
            best_len = len(arr)
            return True
        return False

    g0 = _greedy(deadline, None)
    consider(g0)

    if best is not None:
        ls_result = _local_search(best, deadline)
        consider(ls_result)

    rng = random.Random(2024)
    for _ in range(800):
        if time.monotonic() > deadline:
            break
        g = _greedy(deadline, rng)
        if len(g) > best_len:
            consider(g)
            if best is not None:
                ls_result = _local_search(best, deadline)
                consider(ls_result)
        if best_len >= 60:
            break

    if best is None:
        best = list(range(226, 257))
        if not _check(best):
            best = list(range(1, 31))
    best = [int(v) for v in best]
    return best
