import time
import random
import sys
from math import exp

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Bitmask for differences 1..30
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)

    def is_valid(a):
        diffs = 0
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                d = a[j] - a[i]
                if d > 30:
                    continue
                if diffs & (1 << d):
                    return False
                diffs |= (1 << d)
        return (diffs & target_mask) == target_mask

    # Known candidates (fast path)
    known_candidates = [
        [0, 1, 3, 7, 12, 21, 25],
        [0, 1, 3, 7, 12, 21, 28],
        [0, 1, 3, 7, 12, 25, 28],
        [0, 2, 5, 10, 18, 23, 28],
        [0, 1, 4, 10, 18, 23, 28],
        [0, 1, 2, 7, 12, 22, 28],
        [0, 1, 3, 8, 12, 22, 26],
        [0, 1, 4, 10, 12, 28, 32, 45, 63],
        [0, 1, 4, 10, 12, 28, 32, 45],
        [0, 1, 4, 10, 12, 28, 32, 45, 47],
        [0, 1, 4, 10, 12, 28, 32, 45, 49],
        [0, 1, 4, 10, 12, 18, 28, 32, 45],
        [0, 1, 4, 10, 12, 18, 23, 28, 32],
        [0, 1, 4, 10, 12, 18, 23, 28, 35],
    ]
    for c in known_candidates:
        c = sorted(set(c))
        if 1 <= len(c) <= 32 and is_valid(c):
            return c

    # --- Bitmask DFS: build Sidon set in {0..D} containing 0, covering 1..30 ---
    def dfs_search(D, deadline):
        """Find valid Sidon set in {0..D} with 0, covering 1..30. Return None if not found."""
        found = [None]

        def dfs(current, diff_mask, last):
            if time.time() > deadline:
                return True  # timeout
            if found[0] is not None:
                return True

            # Check completion
            if (diff_mask & target_mask) == target_mask:
                found[0] = list(current)
                return True

            # Pruning: not enough room to cover remaining diffs
            remaining = 0
            for d in range(1, 31):
                if not (diff_mask & (1 << d)):
                    remaining += 1
            # Max new diffs from adding elements in (last, D]
            space = D - last
            if space <= 0:
                return False
            max_new = space * len(current) + space * (space - 1) // 2
            if max_new < remaining:
                return False

            # Try each candidate in increasing order
            for x in range(last + 1, D + 1):
                if time.time() > deadline:
                    return True
                ok = True
                new_mask = diff_mask
                for elem in current:
                    d = x - elem
                    if d > 30:
                        continue
                    if new_mask & (1 << d):
                        ok = False
                        break
                    new_mask |= (1 << d)
                if not ok:
                    continue
                current.append(x)
                if dfs(current, new_mask, x):
                    return True
                current.pop()
            return False

        dfs([0], 0, 0)
        return found[0]

    # Try diameters from lower bound upward
    best_set = None
    for D in range(28, 80):
        if time.time() > deadline - 30:
            break
        result = dfs_search(D, deadline - 30)
        if result is not None:
            best_set = result
            break

    # If DFS didn't find, use greedy fallback
    if best_set is None:
        def greedy():
            a = [0]
            covered = 0
            for d in range(1, 31):
                if covered & (1 << d):
                    continue
                # Try to add element that creates difference d
                found = False
                for x in range(1, 200):
                    if x in a:
                        continue
                    ok = True
                    new_mask = covered
                    for v in a:
                        dd = abs(x - v)
                        if dd > 30:
                            continue
                        if new_mask & (1 << dd):
                            ok = False
                            break
                        new_mask |= (1 << dd)
                    if ok and (new_mask & (1 << d)):
                        a.append(x)
                        a.sort()
                        covered = new_mask
                        found = True
                        break
                if not found:
                    # Force add
                    x = max(a) + d + 1
                    while True:
                        ok = True
                        new_mask = covered
                        for v in a:
                            dd = x - v
                            if dd > 30:
                                continue
                            if new_mask & (1 << dd):
                                ok = False
                                break
                            new_mask |= (1 << dd)
                        if ok:
                            a.append(x)
                            a.sort()
                            covered = new_mask
                            break
                        x += 1
            return a
        best_set = greedy()

    # --- Local optimization: reduce diameter ---
    def local_optimize(a, deadline):
        a = sorted(a)
        best = list(a)
        best_diam = a[-1]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to reduce the last element first (biggest impact)
            for i in range(len(a) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old = a[i]
                # Try replacing with smaller value
                for new_val in range(1, old):
                    if new_val in a:
                        continue
                    trial = list(a)
                    trial[i] = new_val
                    trial.sort()
                    if is_valid(trial) and trial[-1] < best_diam:
                        a = trial
                        best = list(a)
                        best_diam = trial[-1]
                        improved = True
                        break
        return best, best_diam

    if best_set is not None and time.time() < deadline - 5:
        opt, opt_diam = local_optimize(best_set, deadline - 5)
        if opt_diam < best_set[-1]:
            best_set = opt

    # --- ILP-style branch-and-bound polish ---
    # For the best set found, try to find a smaller-diameter solution
    # by systematically exploring which elements to keep and repositioning
    def ilp_polish(a, deadline):
        """Branch-and-bound: try to find valid set with smaller diameter."""
        a = sorted(a)
        best = list(a)
        best_diam = a[-1]

        # Try each possible diameter from current down
        for D in range(best_diam - 1, 27, -1):
            if time.time() > deadline:
                break

            # DFS to find any valid set in {0..D}
            result = dfs_search(D, deadline)
            if result is not None:
                best = result
                best_diam = D
                break

        return best, best_diam

    if best_set is not None and time.time() < deadline - 2:
        polished, polished_diam = ilp_polish(best_set, deadline - 2)
        if polished_diam < best_set[-1]:
            best_set = polished

    return best_set

# EVOLVE-BLOCK-END
