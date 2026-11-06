"""Sidon sets containing 0 realizing differences 1..30, minimizing the diameter."""
import time
import sys

sys.setrecursionlimit(100000)


def construct():
    # Deadline guard: hard wall-clock limit with safety margin.
    deadline = time.time() + 115.0

    # Precompute the required difference set {1..30} as a bitset.
    need_mask = (1 << 31) - 2  # bits 1..30 set

    def is_valid_solution(a):
        # Validate: 0 in set, distinct, all diffs distinct, 1..30 covered.
        if 0 not in a:
            return False
        a = sorted(a)
        if len(a) != len(set(a)):
            return False
        if len(a) > 32:
            return False
        if max(a) > 10**6:
            return False
        diffs = set()
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                d = a[j] - a[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    best = [None]  # best solution found (list of ints)
    best_max = [10**9]

    def update_best(a):
        m = max(a)
        if m < best_max[0]:
            best_max[0] = m
            best[0] = list(a)

    # DFS state:
    #   a: list of chosen elements (sorted)
    #   covered: bitmask of covered differences (1..30)
    #   diff_set: set of all positive differences so far (for conflict check)
    #   max_allowed: current upper bound on largest element (branch-and-bound)

    def dfs(a, covered, diff_set, max_allowed):
        if time.time() > deadline:
            return

        # Objective: minimize the largest element.
        # All required diffs covered and we have a valid set:
        if (covered & need_mask) == need_mask:
            m = a[-1] if a else 0
            update_best(a)
            # Also try to see if we can shrink by not adding more (already minimal set)
            # But we still want the smallest max; record and return (no need to extend)
            return

        # Prune: if current max already >= best_max, and we still need to add elements
        # (which can only increase or keep max), we can prune.
        if a and a[-1] >= best_max[0]:
            return

        # Lower bound on needed max:
        # If we still need difference d_max (largest uncovered), then max must be >= d_max
        # (since 0 is in the set, max - 0 = max, and any diff d requires two elements
        #  differing by d, so max >= d). Actually max must be >= every uncovered d.
        uncovered = need_mask & ~covered
        if uncovered:
            # largest set bit
            lb = uncovered.bit_length() - 1
            if a and a[-1] < lb:
                # We will need to add an element >= lb, so max >= lb
                pass
            # If even the theoretical min max (lb) is >= best_max, prune
            if lb >= best_max[0]:
                return

        # Choose the next element to add.
        # Strategy: the new element x must create at least one new required difference.
        # Candidate x: for each existing element v in a, and each uncovered d,
        #   x = v + d or x = v - d (if >= 0), x not in a, x < max_allowed.
        # Also x could be a fresh large element pairing with another to make a diff,
        # but that's covered by the v+d / v-d generation when we consider all v.

        # To keep branching controlled, generate candidates and sort by x (prefer small).
        cands = set()
        for v in a:
            for d in range(1, 31):
                if not (covered >> d) & 1:  # d uncovered
                    x1 = v + d
                    if 0 <= x1 < max_allowed and x1 not in diff_set:  # placeholder
                        cands.add(x1)
                    x2 = v - d
                    if 0 <= x2 < max_allowed:
                        cands.add(x2)

        # Filter candidates that are already in the set
        a_set = set(a)
        cands = [x for x in cands if x not in a_set and 0 <= x < max_allowed]

        # Sort by x ascending (prefer smaller to keep diameter small)
        cands.sort()

        # Cap candidate count to avoid explosion
        if len(cands) > 200:
            cands = cands[:200]

        for x in cands:
            if time.time() > deadline:
                return

            # Check conflict: x must not create a duplicate difference.
            ok = True
            new_diffs = []
            for v in a:
                d = abs(x - v)
                if d == 0:
                    ok = False
                    break
                if d in diff_set:
                    ok = False
                    break
                new_diffs.append(d)
            if not ok:
                continue

            new_covered = covered
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_covered |= (1 << d)

            # Only proceed if we gained at least one new required difference
            if new_covered == covered:
                continue

            new_diff_set = diff_set | set(new_diffs)
            new_max_allowed = min(max_allowed, best_max[0])  # keep bound

            # Recurse
            dfs(a + [x], new_covered, new_diff_set, new_max_allowed)

            # Update bound after this branch
            if best_max[0] <= x:
                # Further x in this loop are >= x >= best_max, can break
                break

    # Initial: a = [0], covered = 0, diff_set = empty, max_allowed = 10**6
    # But we can set a tighter initial bound. The theoretical min diameter for
    # realizing diffs 1..30: need max >= 30. Start with a reasonable bound.
    # Use a heuristic initial bound: start high, let branch-and-bound tighten.
    initial_max = 10**6

    dfs([0], 0, set(), initial_max)

    if best[0] is not None:
        # Validate before returning
        if is_valid_solution(best[0]):
            return best[0]

    # Fallback: return a valid (possibly non-optimal) solution.
    # Use the seed's greedy as a safety net.
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        placed = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                placed = True
                break
        if not placed:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
