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
        # Check: new sums for each k must not collide with sums of other cardinalities
        # acc_before = union of all prev[k]
        acc_before = self.acc
        for k in range(1, new_max_k + 1):
            new_sums = ndp[k] & ~prev[k]
            # new_sums must not intersect with any other cardinality's sums
            # other_sums = acc_before ^ prev[k] (sums not in cardinality k)
            # But careful: acc_before includes prev[k], so other = acc_before & ~prev[k]
            other = acc_before & ~prev[k]
            if new_sums & other:
                self.valid = False
                return False
        for k in range(1, new_max_k + 1):
            self.acc |= ndp[k]
        self.dp = ndp
        self.max_k = new_max_k
        return True

    def snapshot(self):
        """Return a copy of the verifier state for rollback."""
        v = Verifier(self.n)
        v.dp = list(self.dp)
        v.acc = self.acc
        v.valid = self.valid
        v.max_k = self.max_k
        return v

    def restore(self, snap):
        self.dp = list(snap.dp)
        self.acc = snap.acc
        self.valid = snap.valid
        self.max_k = snap.max_k


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


def _try_remove(arr, used, ver, idx):
    """Remove element at idx and rebuild verifier from remaining elements."""
    removed = arr.pop(idx)
    used.discard(removed)
    ver2 = Verifier(60)
    for v in arr:
        ver2.add(v)
    return removed, ver2


def _greedy(deadline, rng, mode="desc"):
    cur = []
    used = set()
    ver = Verifier(60)

    if mode == "desc":
        order = list(range(256, 0, -1))
    elif mode == "asc":
        order = list(range(1, 257))
    elif mode == "mid":
        order = list(range(128, 0, -1)) + list(range(256, 128, -1))
    else:
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

    # Try adding remaining values in ascending order to fill gaps
    for v in range(1, 257):
        if time.monotonic() > deadline:
            break
        if v not in used:
            _try_add(cur, used, ver, v)

    return cur


def _local_search(arr, deadline):
    if len(arr) >= 60:
        return arr
    cur = list(arr)
    used = set(cur)
    ver = Verifier(60)
    for v in cur:
        ver.add(v)

    # Pass A: 1-for-2 (remove one, try to add two)
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

    # Pass C: 2-for-3 (remove two, try to add three)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for i in range(n):
            if time.monotonic() > deadline:
                break
            for j in range(i + 1, n):
                if time.monotonic() > deadline:
                    break
                r1 = cur[i]
                r2 = cur[j]
                # Remove both
                if j > i:
                    cur.pop(j)
                    cur.pop(i)
                else:
                    cur.pop(i)
                    cur.pop(j)
                used.discard(r1)
                used.discard(r2)
                ver2 = Verifier(60)
                for v in cur:
                    ver2.add(v)
                added_count = 0
                for v in range(256, 0, -1):
                    if v not in used and _try_add(cur, used, ver2, v):
                        added_count += 1
                        if added_count == 3:
                            break
                    if time.monotonic() > deadline:
                        break
                if added_count >= 3:
                    improved = True
                    break
                else:
                    for _ in range(added_count):
                        cur.pop()
                    cur.insert(i, r1)
                    cur.insert(j, r2)
                    used.add(r1)
                    used.add(r2)
            if improved:
                break

    # Final: try to add more
    for v in range(256, 0, -1):
        if time.monotonic() > deadline:
            break
        if v not in used:
            _try_add(cur, used, ver, v)

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

    # Try multiple greedy strategies
    g0 = _greedy(deadline, None, "desc")
    consider(g0)

    g1 = _greedy(deadline, None, "asc")
    consider(g1)

    g2 = _greedy(deadline, None, "mid")
    consider(g2)

    # Local search on best
    if best is not None and best_len < 60:
        ls_result = _local_search(best, deadline)
        consider(ls_result)

    # Random restarts with different modes
    rng = random.Random(2024)
    modes = ["desc", "asc", "mid"]
    for _ in range(600):
        if time.monotonic() > deadline:
            break
        mode = modes[_ % len(modes)]
        g = _greedy(deadline, rng, mode)
        if len(g) > best_len:
            consider(g)
            if best is not None and best_len < 60:
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
