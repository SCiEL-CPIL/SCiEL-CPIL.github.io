"""Sidon set (Golomb ruler) covering differences 1..30; minimize max element."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0   # safety margin before 120s kill
    TARGET = 30
    MAX_ELEM = 10**6
    best = {"val": None}

    # Global search state for pruning
    def solve():
        # S is a sorted tuple; used_diffs is a set of all pairwise positive diffs
        # covered is a bitmask of which of 1..30 are present
        # We do a DFS trying to extend S with the smallest feasible new element.
        # We keep the smallest complete solution seen.
        pass

    # We'll implement an iterative DFS with explicit stack to control the deadline.
    # Node: (S_tuple, used_diffs_set, covered_mask, next_min)
    # But to keep it simple and fast, do a recursive DFS with memo-free pruning
    # and a time check.

    import sys
    sys.setrecursionlimit(10000)

    def all_diffs(S):
        return [S[j] - S[i] for i in range(len(S)) for j in range(i + 1, len(S))]

    def is_valid_add(S, x, used_diffs):
        # returns new used_diffs if x can be added without collision, else None
        new_diffs = []
        for v in S:
            d = abs(x - v)
            if d in used_diffs:
                return None
            new_diffs.append(d)
        return new_diffs

    def dfs(S, used_diffs, covered_mask, depth):
        # S: sorted tuple
        if time.time() > deadline:
            return
        if covered_mask == (1 << (TARGET + 1)) - 2:  # bits 1..30 set
            val = S[-1]
            if best["val"] is None or val < best["val"]:
                best["val"] = val
                return
        # Branch: try adding a new element x > S[-1]
        # Candidate x range: from S[-1]+1 up to some bound
        # Bound: we need to still be able to realize missing diffs; use a loose bound
        # but prune by current best.
        upper = best["val"] if best["val"] is not None else 2 * S[-1] + TARGET + 5
        # Try x in increasing order
        for x in range(S[-1] + 1, upper + 1):
            if time.time() > deadline:
                return
            new_diffs = is_valid_add(S, x, used_diffs)
            if new_diffs is None:
                continue
            new_used = set(used_diffs)
            for d in new_diffs:
                new_used.add(d)
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= TARGET:
                    new_mask |= (1 << d)
            newS = S + (x,)
            dfs(newS, new_used, new_mask, depth + 1)
            if best["val"] is not None and best["val"] <= x:
                # can't improve with larger x in this branch beyond current best
                # but other branches at this level with smaller x already tried;
                # since x increases, and best <= x, larger x won't help here
                break

    # Initial: S = (0,)
    dfs((0,), set(), 0, 0)

    if best["val"] is None:
        # Fallback: greedy construction (guaranteed valid)
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            placed = False
            for v in sorted(a):
                for sgn in (d, -d):
                    x = v + sgn
                    if x < 0 or x in a:
                        continue
                    trial = sorted(a + [x])
                    dd = diffs(trial)
                    if len(set(dd)) == len(dd):
                        a = trial
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    dd = diffs(trial)
                    if len(set(dd)) == len(dd):
                        a = trial
                        break
                    y += 1
        return a

    # Reconstruct: we only stored best value, not the set. Re-run DFS to capture the set.
    # Simpler: redo with set capture.
    return _construct_with_capture(best["val"])

def _construct_with_capture(target_max):
    TARGET = 30
    def is_valid_add(S, x, used_diffs):
        for v in S:
            if abs(x - v) in used_diffs:
                return None
        return True
    def dfs(S, used_diffs, covered_mask):
        if covered_mask == (1 << (TARGET + 1)) - 2:
            if S[-1] <= target_max:
                return S
            return None
        upper = target_max
        for x in range(S[-1] + 1, upper + 1):
            if not is_valid_add(S, x, used_diffs):
                continue
            new_used = set(used_diffs)
            new_mask = covered_mask
            for v in S:
                d = x - v
                new_used.add(d)
                if 1 <= d <= TARGET:
                    new_mask |= (1 << d)
            res = dfs(S + (x,), new_used, new_mask)
            if res is not None:
                return res
        return None
    r = dfs((0,), set(), 0)
    if r is None:
        # Should not happen; return a trivial valid set via greedy
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            placed = False
            for v in sorted(a):
                for sgn in (d, -d):
                    x = v + sgn
                    if x < 0 or x in a:
                        continue
                    trial = sorted(a + [x])
                    dd = diffs(trial)
                    if len(set(dd)) == len(dd):
                        a = trial
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    dd = diffs(trial)
                    if len(set(dd)) == len(dd):
                        a = trial
                        break
                    y += 1
        return a
    return list(r)

# EVOLVE-BLOCK-END
