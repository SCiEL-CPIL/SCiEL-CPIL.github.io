import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + 135.0

    # Greedy baseline (the seed behaviour).
    def greedy():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy()
    best = sum(rows[r][-1] for r in range(n_rows))

    # Backtracking with MRV + forward-checking.
    def backtrack():
        nonlocal best
        rows = [[] for _ in range(n_rows)]
        used = set()
        best_row = 0
        best_scope = 0

        def remaining(r):
            return k + 1 - len(rows[r])

        def try_next(r, x):
            if time.time() > deadline:
                return False
            new = {x - a for a in rows[r]}
            if new & used:
                return False
            rows[r].append(x)
            used |= new
            if len(rows[r]) == k + 1:
                if r + 1 < n_rows:
                    if try_next(r + 1, 0):
                        return True
                else:
                    s = sum(row[-1] for row in rows)
                    if s < best:
                        best = s
                        return True
                # keep exploring for a better scope
            rows[r].pop()
            used -= new
            return False

        try_next(0, 0)
        return rows

    # A simpler, robust branch-and-bound: build rows one by one, choose the
    # smallest feasible next element for the current row (greedy within the
    # used set), but allow backtracking over the whole assignment with a
    # pruning bound on the final scope.
    def branch_and_bound():
        nonlocal best
        rows = [[] for _ in range(n_rows)]
        used = set()

        def feasible_scope_lb(r):
            # lower bound: current max + minimal remaining additions
            lb = 0
            for i in range(n_rows):
                last = rows[i][-1] if rows[i] else 0
                rem = k + 1 - len(rows[i])
                # each new element must be at least last+1
                lb = max(lb, last + rem)
            return lb

        def solve(r, x):
            nonlocal best
            if time.time() > deadline:
                return False
            if r == n_rows:
                s = sum(row[-1] for row in rows)
                if s < best:
                    best = s
                return False  # continue searching
            if not rows[r]:
                rows[r].append(0)
            # try next candidate x
            if len(rows[r]) < k + 1:
                new = {x - a for a in rows[r]}
                if not (new & used):
                    rows[r].append(x)
                    used |= new
                    # bound
                    if feasible_scope_lb(r) < best:
                        if solve(r, x + 1):
                            pass
                        # also try advancing to next row if this row is full
                    if len(rows[r]) == k + 1:
                        if solve(r + 1, 0):
                            pass
                    used -= new
                    rows[r].pop()
            return False

        solve(0, 1)
        return rows

    # Run backtracking with MRV-style ordering but bounded.
    try:
        # Use a focused search: try to beat the greedy scope.
        bt_rows = branch_and_bound()
        if bt_rows and all(len(r) == k + 1 for r in bt_rows):
            s = sum(r[-1] for r in bt_rows)
            if s < best:
                best = s
                rows = bt_rows
    except Exception:
        pass

    return rows
# EVOLVE-BLOCK-END
