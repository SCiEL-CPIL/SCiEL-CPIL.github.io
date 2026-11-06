"""Optimized Sidon set construction combining fast diff-checking with known sets and local search."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def check(a):
        """Fast incremental Sidon check. Returns (ok, diff_set).
        ok is True iff a (sorted) is a Sidon set; diff_set is the set of positive diffs."""
        diffs = set()
        n = len(a)
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                d = a[j] - ai
                if d in diffs:
                    return False, None
                diffs.add(d)
        return True, diffs

    def is_sidon(a):
        return check(a)[0]

    def diffs_of(a):
        a = sorted(a)
        return set(a[j] - a[i] for i in range(len(a)) for j in range(i + 1, len(a)))

    # --- Fast greedy construction (seed strategy) ---
    def greedy_construct():
        a = [0]
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            # Try to add x = v +/- d for some existing v, smallest x first
            candidates = sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a))
            found = False
            for x in candidates:
                trial = a + [x]
                ok, nd = check(trial)
                if ok and d in nd:
                    a = sorted(trial)
                    covered = nd
                    found = True
                    break
            if not found:
                # Fallback: add two elements far away to realize d
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = a + [y, y + d]
                    ok, nd = check(trial)
                    if ok:
                        a = sorted(trial)
                        covered = nd
                        break
                    y += 1
        return a

    # --- Try known difference sets first (cheap shortcut) ---
    known_candidates = [
        # (7,7,1) perfect difference sets
        [0, 1, 3, 7, 12, 21, 25],
        [0, 1, 3, 7, 12, 21, 28],
        [0, 1, 3, 7, 12, 25, 28],
        [0, 2, 5, 10, 18, 23, 28],
        [0, 1, 4, 10, 18, 23, 28],
        [0, 1, 2, 7, 12, 22, 28],
        [0, 1, 3, 8, 12, 22, 26],
        # (9,9,1) perfect difference set with diameter 63
        [0, 1, 4, 10, 12, 28, 32, 45, 63],
        # Other known Sidon sets with small diameter
        [0, 1, 4, 10, 12, 28, 32, 45],
        [0, 1, 4, 10, 12, 28, 32, 45, 47],
        [0, 1, 4, 10, 12, 28, 32, 45, 49],
    ]
    for c in known_candidates:
        c = sorted(set(c))
        if len(c) < 1 or len(c) > 32:
            continue
        ok, cd = check(c)
        if ok and all(d in cd for d in range(1, 31)):
            return c

    # --- Build initial solution via greedy ---
    best = greedy_construct()
    best_max = max(best)

    # --- Local search: try to reduce the max element ---
    if time.time() < deadline - 5:
        random.seed(42)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            cur = list(best)
            
            # Strategy 1: Try replacing the largest element with a smaller unused value
            max_val = max(cur)
            max_idx = cur.index(max_val)
            for new_val in range(1, max_val):
                if new_val in cur:
                    continue
                trial = cur[:max_idx] + [new_val] + cur[max_idx + 1:]
                trial = sorted(trial)
                if len(set(trial)) != len(trial):
                    continue
                ok, td = check(trial)
                if ok and all(d in td for d in range(1, 31)):
                    if max(trial) < best_max:
                        best = trial
                        best_max = max(trial)
                        improved = True
                        break
            
            # Strategy 2: Try removing an element and replacing it with a smaller value
            if not improved:
                for idx in range(len(cur)):
                    old_val = cur[idx]
                    for new_val in range(1, best_max):
                        if new_val in cur:
                            continue
                        trial = cur[:idx] + [new_val] + cur[idx + 1:]
                        trial = sorted(trial)
                        if len(set(trial)) != len(trial):
                            continue
                        ok, td = check(trial)
                        if ok and all(d in td for d in range(1, 31)):
                            if max(trial) < best_max:
                                best = trial
                                best_max = max(trial)
                                improved = True
                                break
                    if improved:
                        break
            
            # Strategy 3: Try removing the largest element entirely (if we have >7 elements)
            if not improved and len(cur) > 7:
                max_val = max(cur)
                max_idx = cur.index(max_val)
                trial = cur[:max_idx] + cur[max_idx + 1:]
                trial = sorted(trial)
                if len(trial) >= 1:
                    ok, td = check(trial)
                    if ok and all(d in td for d in range(1, 31)):
                        if max(trial) < best_max:
                            best = trial
                            best_max = max(trial)
                            improved = True
            
            # Strategy 4: Try swapping two elements to get a smaller max
            if not improved and len(cur) >= 2:
                for i in range(len(cur) - 1):
                    for j in range(i + 1, len(cur)):
                        if cur[i] == 0 or cur[j] == 0:
                            continue
                        # Try swapping values
                        trial = list(cur)
                        trial[i], trial[j] = trial[j], trial[i]
                        trial = sorted(trial)
                        if len(set(trial)) != len(trial):
                            continue
                        ok, td = check(trial)
                        if ok and all(d in td for d in range(1, 31)):
                            if max(trial) < best_max:
                                best = trial
                                best_max = max(trial)
                                improved = True
                                break
                    if improved:
                        break
    return best
# EVOLVE-BLOCK-END
