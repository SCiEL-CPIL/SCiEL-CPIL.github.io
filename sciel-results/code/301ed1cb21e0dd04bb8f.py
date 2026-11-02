import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Hybrid: greedy init, SA with row-replacement (backtracking row solver)
    and single-position moves, plus a deterministic polish pass.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # explicit guard: 10s margin before 150s kill

    def _remaining():
        return deadline - time.time()

    # ---- Helpers ----
    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ri)
        return diffs

    def _validate(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for d in _row_diffs(row):
                if d in used:
                    return False
                used.add(d)
        return True

    def _scope(rows):
        return max(r[-1] for r in rows)

    # ---- Greedy construction ----
    def greedy_construct(seed=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # ---- Backtracking row solver ----
    def solve_row(forbidden, max_val, node_budget=40000):
        """Return a valid row (k+1 ints, starts 0, increasing) with diffs avoiding
        `forbidden` and internally distinct, last entry <= max_val; else None."""
        row = [0]
        used_in_row = set()
        nodes = [0]

        def rec(pos, last):
            if nodes[0] > node_budget:
                return False
            if pos == k + 1:
                return True
            nodes[0] += 1
            lo = last + 1
            hi = max_val
            if hi < lo:
                return False
            cands = []
            for x in range(lo, hi + 1):
                new_diffs = {x - a for a in row}
                if len(new_diffs) != len(row):
                    continue
                if new_diffs & forbidden:
                    continue
                if new_diffs & used_in_row:
                    continue
                cands.append(x)
                if len(cands) >= 80:
                    break
            if not cands:
                return False
            random.shuffle(cands)
            for x in cands:
                row.append(x)
                nd = {x - a for a in row[:-1]}
                used_in_row |= nd
                if rec(pos + 1, x):
                    return True
                row.pop()
                used_in_row -= nd
            return False

        if rec(1, 0):
            return row[:]
        return None

    # ---- Local search: SA with row-replacement + single-position moves ----
    def local_search2(rows, deadline):
        cur = [r[:] for r in rows]
        cur_diffs = [_row_diffs(r) for r in cur]
        cur_scope = _scope(cur)
        best = [r[:] for r in cur]
        best_scope = cur_scope

        T = 50.0
        T_min = 0.5
        cool = 0.9998
        no_improve = 0
        max_no_improve = 30000
        restarts = 0
        max_restarts = 60

        def other_diffs(exclude):
            s = set()
            for i in range(n_rows):
                if i != exclude:
                    s |= cur_diffs[i]
            return s

        def propose(ri):
            if random.random() < 0.65:
                forbidden = other_diffs(ri)
                if random.random() < 0.8:
                    max_val = cur_scope
                else:
                    max_val = cur_scope + random.randint(5, 40)
                new_row = solve_row(forbidden, max_val, node_budget=25000)
                if new_row is None:
                    return None
                return new_row, _row_diffs(new_row)
            else:
                pi = random.randrange(1, k + 1)
                old_val = cur[ri][pi]
                low = cur[ri][pi - 1] + 1
                high = (cur[ri][pi + 1] - 1) if pi < k else old_val + 30
                if high < low:
                    return None
                for _ in range(10):
                    if random.random() < 0.6:
                        new_val = random.randint(low, old_val)
                    else:
                        new_val = random.randint(old_val, high)
                    if new_val <= cur[ri][pi - 1]:
                        continue
                    if pi < k and new_val >= cur[ri][pi + 1]:
                        continue
                    cand = cur[ri][:]
                    cand[pi] = new_val
                    nd = _row_diffs(cand)
                    if len(nd) != k * (k + 1) // 2:
                        continue
                    forbidden = other_diffs(ri)
                    if nd & forbidden:
                        continue
                    return cand, nd
                return None

        while _remaining() > 3.0:
            ri = random.randrange(n_rows)
            proposed = propose(ri)
            if proposed is None:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            new_row, new_diffs = proposed
            old_row = cur[ri]
            old_diffs = cur_diffs[ri]
            cur[ri] = new_row
            cur_diffs[ri] = new_diffs
            new_scope = _scope(cur)
            delta = new_scope - cur_scope
            cur_scope = new_scope

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.4
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.5))
                    accept = random.random() < prob

            if not accept:
                cur[ri] = old_row
                cur_diffs[ri] = old_diffs
                cur_scope = _scope(cur)
            else:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in cur]
                    no_improve = 0
                else:
                    no_improve += 1

            T = max(T_min, T * cool)

            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                cur = [r[:] for r in best]
                cur_diffs = [_row_diffs(r) for r in cur]
                cur_scope = best_scope
                T = max(40.0, T * 1.5)
                no_improve = 0

        return best, best_scope

    # ---- Deterministic polish: greedily shrink entries without increasing scope ----
    def polish(rows, deadline):
        cur = [r[:] for r in rows]
        cur_diffs = [_row_diffs(r) for r in cur]
        cur_scope = _scope(cur)
        improved = True
        while improved and _remaining() > 2.0:
            improved = False
            for ri in range(n_rows):
                for pi in range(1, k + 1):
                    if _remaining() < 2.0:
                        break
                    old_val = cur[ri][pi]
                    low = cur[ri][pi - 1] + 1
                    for new_val in range(old_val - 1, low - 1, -1):
                        if pi < k and new_val >= cur[ri][pi + 1]:
                            continue
                        cand = cur[ri][:]
                        cand[pi] = new_val
                        nd = _row_diffs(cand)
                        if len(nd) != k * (k + 1) // 2:
                            continue
                        forbidden = set()
                        for i in range(n_rows):
                            if i != ri:
                                forbidden |= cur_diffs[i]
                        if nd & forbidden:
                            continue
                        # tentatively apply
                        old_row = cur[ri]
                        old_diffs = cur_diffs[ri]
                        cur[ri] = cand
                        cur_diffs[ri] = nd
                        new_scope = _scope(cur)
                        if new_scope <= cur_scope:
                            cur_scope = new_scope
                            improved = True
                            break
                        else:
                            # revert
                            cur[ri] = old_row
                            cur_diffs[ri] = old_diffs
                            break
        if _scope(cur) <= _scope(rows):
            return cur, _scope(cur)
        return rows, _scope(rows)

    # ---- Main driver ----
    best_solution = greedy_construct()
    best_scope = _scope(best_solution)

    # A few randomized greedy inits for diversity
    for seed in (1, 7, 42):
        if _remaining() < 10.0:
            break
        cand = greedy_construct()
        if _scope(cand) < best_scope:
            best_scope = _scope(cand)
            best_solution = cand

    # Main SA with row-replacement
    sa_rows, sa_scope = local_search2(best_solution, deadline)
    if sa_scope < best_scope:
        best_scope = sa_scope
        best_solution = sa_rows

    # Polish
    if _remaining() > 5.0:
        pol_rows, pol_scope = polish(best_solution, deadline)
        if pol_scope < best_scope:
            best_scope = pol_scope
            best_solution = pol_rows

    # Final validation; fall back to greedy if broken
    if not _validate(best_solution):
        best_solution = greedy_construct()
        best_scope = _scope(best_solution)

    return best_solution
# EVOLVE-BLOCK-END
