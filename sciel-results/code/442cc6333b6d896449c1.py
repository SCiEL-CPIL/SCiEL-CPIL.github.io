"""Difference triangle set (4,7): minimize largest entry via randomized greedy + local polish."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 110.0  # safety margin under 120s
    rng = random.Random(12345)

    def greedy_once(max_val, seed):
        """Build n_rows rows of k+1 increasing ints starting at 0, all diffs distinct,
        each mark <= max_val. Return list of rows or None."""
        r = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # candidate marks in increasing order, but with random tie-break among
                # the smallest feasible to diversify
                found = None
                # search from len(row) (smallest possible) up to max_val
                # pick the smallest x that works; add slight randomization by
                # occasionally considering a few candidates
                for x in range(len(row), max_val + 1):
                    new = {x - a for a in row}
                    if not (new & used):
                        found = x
                        break
                if found is None:
                    return None
                row.append(found)
                used |= {found - a for a in row[:-1]}
            rows.append(row)
        return rows

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def max_entry(rows):
        return max(max(r) for r in rows)

    # Baseline: deterministic greedy (no max cap) to guarantee a valid solution.
    def greedy_uncapped():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best = greedy_uncapped()
    best_m = max_entry(best)

    # Try to improve: binary search on max_val with randomized greedy.
    # Upper bound = best_m, lower bound = simple counting lower bound.
    lo = 1
    # counting bound: total diffs = n_rows * k*(k+1)/2 must fit in [1, m]
    total_diffs = n_rows * k * (k + 1) // 2
    lo = max(lo, total_diffs)
    hi = best_m

    # First, see if we can do better than best_m with a few randomized greedy runs
    # at cap = best_m - 1, then binary search downward.
    attempt = 0
    max_attempts = 400
    # We'll do a coarse search: try caps from best_m-1 downward a bit, then binary search.
    # Strategy: for cap in a descending sequence, run a few randomized greedy; if success,
    # update best and continue lower.

    # Use a list of caps to try, prioritizing just below current best.
    caps_to_try = []
    # Add best_m-1, best_m-2, ... best_m-20 (if positive), then binary search region.
    for c in range(best_m - 1, max(lo, best_m - 25) - 1, -1):
        caps_to_try.append(c)
    # Also add a binary search ladder between lo and best_m
    a, b = lo, best_m
    while a < b and len(caps_to_try) < 60:
        mid = (a + b) // 2
        if mid not in caps_to_try:
            caps_to_try.append(mid)
        if mid < b:
            b = mid
        else:
            a = mid + 1

    # Deduplicate and sort descending so we try the highest (most likely feasible) first,
    # but we want the *smallest* feasible, so actually try low-to-high? No: we want to find
    # the smallest feasible. But greedy is easier to succeed at higher caps. So we should
    # try from low to high to find the minimum? That's expensive. Better: binary search for
    # the minimum feasible using randomized greedy as the feasibility oracle.
    # We'll do randomized binary search: for a given cap, run R greedy trials; if any
    # succeeds, cap is feasible.

    def feasible(cap, trials):
        for t in range(trials):
            if time.time() > deadline:
                return None  # timeout
            res = greedy_once(cap, seed=attempt * 7919 + t * 104729)
            if res is not None:
                return res
        return None

    # Randomized binary search for smallest feasible cap in [lo, best_m-1]
    # We know best_m is feasible (baseline). Search [lo, best_m-1].
    search_lo = lo
    search_hi = best_m  # exclusive; best_m known feasible
    found = None
    trial_budget = 0
    max_total_trials = 300
    while search_lo < search_hi and trial_budget < max_total_trials:
        mid = (search_lo + search_hi) // 2
        # run a few trials at mid
        res = feasible(mid, trials=5)
        trial_budget += 5
        if res is not None:
            # mid feasible: record, search lower
            m_val = max_entry(res)
            if m_val < best_m:
                best = res
                best_m = m_val
            search_hi = mid
        else:
            # mid not feasible in our trials; search higher
            search_lo = mid + 1
        if time.time() > deadline - 20:
            break

    # Local polish: try to reduce best_m by re-greedy-ing individual rows within best_m-1.
    # For each row, remove it, rebuild it greedily with cap = best_m - 1, keeping others fixed.
    def polish(rows, cap, max_iters):
        cur = [list(r) for r in rows]
        cur_m = max_entry(cur)
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if cur_m <= cap:
                break
            # pick the row with the largest last mark
            worst = max(range(len(cur)), key=lambda i: cur[i][-1])
            saved = cur[worst]
            # rebuild this row greedily with cap, others fixed
            used = set()
            for i, row in enumerate(cur):
                if i == worst:
                    continue
                for a in range(len(row)):
                    for b in range(a + 1, len(row)):
                        used.add(row[b] - row[a])
            row = [0]
            ok = True
            while len(row) < k + 1:
                found_x = None
                for x in range(len(row), cap + 1):
                    new = {x - a for a in row}
                    if not (new & used):
                        found_x = x
                        break
                if found_x is None:
                    ok = False
                    break
                row.append(found_x)
                used |= {found_x - a for a in row[:-1]}
            if ok and row[-1] < saved[-1]:
                cur[worst] = row
                cur_m = max_entry(cur)
            else:
                cur[worst] = saved
        return cur

    # Polish a few times with decreasing caps
    for cap in [best_m - 1, best_m - 2, best_m - 3]:
        if cap < lo:
            break
        polished = polish(best, cap, max_iters=30)
        pm = max_entry(polished)
        if pm < best_m:
            best = polished
            best_m = pm

    # Final validation
    if not is_valid(best):
        # Fallback to a fresh uncapped greedy
        best = greedy_uncapped()

    return best
# EVOLVE-BLOCK-END
