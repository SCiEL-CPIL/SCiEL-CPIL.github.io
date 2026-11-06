"""Search for a small-diameter Sidon set (B_2) containing 0
whose positive differences include 1..30. Minimizes the largest element."""
import time
import random
import sys

_DEADLINE_FRACTION = 0.88
_WALL_LIMIT = 150.0


def _is_valid(a):
    s = sorted(set(a))
    if 0 not in s:
        return False
    if len(s) > 32:
        return False
    if max(s) > 10**6:
        return False
    diffs = []
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(s[j] - s[i])
    if len(set(diffs)) != len(diffs):
        return False
    dset = set(diffs)
    for d in range(1, 31):
        if d not in dset:
            return False
    return True


def _covers(a):
    s = sorted(a)
    n = len(s)
    dset = set()
    for i in range(n):
        for j in range(i + 1, n):
            dset.add(s[j] - s[i])
    return all(d in dset for d in range(1, 31))


def _is_sidon(a):
    s = sorted(a)
    n = len(s)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(s[j] - s[i])
    return len(set(diffs)) == len(diffs)


def _diffs_of(a):
    s = sorted(a)
    n = len(s)
    return [s[j] - s[i] for i in range(n) for j in range(i + 1, n)]


def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT * _DEADLINE_FRACTION

    best = None
    best_max = 10**6

    def consider(a):
        nonlocal best, best_max
        a = sorted(set(a))
        if _is_valid(a):
            m = max(a)
            if m < best_max:
                best_max = m
                best = list(a)

    # ---- 1. Greedy baseline ----
    def greedy_baseline():
        a = [0]
        used = set()
        for d in range(1, 31):
            if d in used:
                continue
            placed = False
            cands = []
            for v in a:
                for x in (v + d, v - d):
                    if x >= 0 and x not in a:
                        cands.append(x)
            cands = sorted(set(cands))
            for x in cands:
                newdiffs = [abs(x - v) for v in a]
                if len(set(newdiffs)) != len(newdiffs):
                    continue
                if any(nd in used for nd in newdiffs):
                    continue
                a.append(x)
                used.update(newdiffs)
                placed = True
                break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = a + [y, y + d]
                    newdiffs = []
                    for v in a:
                        newdiffs.append(y - v)
                        newdiffs.append(y + d - v)
                    newdiffs.append(d)
                    if len(set(newdiffs)) != len(newdiffs):
                        y += 1
                        continue
                    if any(nd in used for nd in newdiffs):
                        y += 1
                        continue
                    a = sorted(a + [y, y + d])
                    used.update(newdiffs)
                    break
        return a

    baseline = greedy_baseline()
    consider(baseline)

    # ---- 2. Backtracking search with pruning ----
    # Strategy: build a Sidon set containing 0, covering 1..30, minimizing max element.
    # We use a priority-queue-like approach: try to add elements that cover the most
    # uncovered differences while keeping the set small and within a diameter bound.

    def backtrack_search(D, max_size=16, iters=200000):
        """Try to find a Sidon set within [0, D] covering 1..30, size <= max_size."""
        if time.time() > deadline:
            return None

        # We'll use a randomized backtracking approach.
        # Start with 0, then try to add elements.
        # Key insight: we need to cover differences 1..30.
        # A Sidon set of size k has k*(k-1)/2 differences, all distinct.
        # To cover 1..30, we need at least 30 differences, so k*(k-1)/2 >= 30 => k >= 9.

        rng_local = random.Random(42 + D)

        for attempt in range(iters):
            if time.time() > deadline:
                return None

            # Randomized greedy with backtracking
            a = [0]
            used = set()
            covered = set()

            # Determine which differences are still needed
            needed = set(range(1, 31))

            # Build up the set
            success = True
            while needed:
                if len(a) >= max_size:
                    success = False
                    break

                # Find candidate positions
                cands = []
                for v in a:
                    for x in (v + 1, v - 1):
                        if 0 <= x <= D and x not in a:
                            cands.append(x)
                    for d in list(needed):
                        for x in (v + d, v - d):
                            if 0 <= x <= D and x not in a:
                                cands.append(x)

                if not cands:
                    # Try random positions
                    for _ in range(10):
                        x = rng_local.randint(1, D)
                        if x not in a:
                            cands.append(x)

                if not cands:
                    success = False
                    break

                # Score candidates by how many new differences they cover
                scored = []
                for x in cands:
                    newdiffs = [abs(x - v) for v in a]
                    if len(set(newdiffs)) != len(newdiffs):
                        continue
                    if any(nd in used for nd in newdiffs):
                        continue
                    new_covered = sum(1 for nd in newdiffs if nd in needed)
                    scored.append((new_covered, x))

                if not scored:
                    success = False
                    break

                # Pick the best candidate (most new coverage)
                scored.sort(key=lambda t: (-t[0], t[1]))
                x = scored[0][1]

                newdiffs = [abs(x - v) for v in a]
                a.append(x)
                used.update(newdiffs)
                needed -= set(newdiffs)

            if success and len(a) <= max_size:
                a = sorted(set(a))
                if _is_sidon(a) and _covers(a) and 0 in a:
                    return a

        return None

    # Try various diameter bounds
    for D in range(25, 120):
        if time.time() > deadline:
            break
        for k in (9, 10, 11, 12, 13, 14, 15, 16):
            if time.time() > deadline:
                break
            res = backtrack_search(D, k, iters=5000)
            if res is not None:
                consider(res)
                if best_max <= D:
                    break

    # ---- 3. Local search: shrink the max element ----
    if best is not None:
        if time.time() < deadline:
            rng = random.Random(99999)

            def local_shrink(a, iters=20000):
                cur = sorted(a)
                cur_max = max(cur)
                for _ in range(iters):
                    if time.time() > deadline:
                        break
                    idx = rng.randint(0, len(cur) - 1)
                    if cur[idx] == 0:
                        continue
                    old = cur[idx]
                    if old <= 1:
                        continue
                    new = rng.randint(1, old - 1)
                    if new in cur:
                        continue
                    trial = cur[:idx] + [new] + cur[idx + 1:]
                    trial = sorted(set(trial))
                    if len(trial) != len(cur):
                        continue
                    if not _is_sidon(trial):
                        continue
                    if not _covers(trial):
                        continue
                    if max(trial) < cur_max:
                        cur = trial
                        cur_max = max(cur)
                return cur

            shrunk = local_shrink(best)
            consider(shrunk)

            # Additional: try to remove redundant elements
            if time.time() < deadline:
                def try_remove_elements(a):
                    cur = sorted(a)
                    improved = True
                    while improved:
                        improved = False
                        if time.time() > deadline:
                            break
                        for i in range(len(cur)):
                            if cur[i] == 0:
                                continue
                            trial = cur[:i] + cur[i+1:]
                            if len(trial) < 2:
                                continue
                            if _is_sidon(trial) and _covers(trial) and 0 in trial:
                                cur = trial
                                improved = True
                                break
                    return cur

                reduced = try_remove_elements(best)
                consider(reduced)

    if best is None:
        return baseline
    return best


if __name__ == "__main__":
    print(construct())
