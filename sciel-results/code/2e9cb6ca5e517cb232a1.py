"""Difference triangle set solver: CSP backtracking with forward checking."""
import time

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 110.0
    total_diffs = n_rows * (k * (k + 1) // 2)

    # Fast greedy baseline (also serves as a fallback / seed ordering)
    best = _greedy(n_rows, k)
    best_scope = _scope(best)

    # Try CSP backtracking with different heuristics
    for order in ("min", "max", "alt"):
        if time.time() >= deadline:
            break
        res = _csp(n_rows, k, total_diffs, deadline, order)
        if res is not None:
            s = _scope(res)
            if s < best_scope:
                best, best_scope = res, s

    # Final validation
    if not _valid(best, n_rows, k):
        # reconstruct a guaranteed-valid greedy (should not happen)
        best = _greedy(n_rows, k)
    return best


def _scope(rows):
    m = 0
    for r in rows:
        if r:
            m = max(m, r[-1])
    return m


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(len(r)):
            if i and r[i] <= r[i - 1]:
                return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _greedy(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _csp(n_rows, k, total_diffs, deadline, order):
    used = [False] * (total_diffs + 1)
    rows = []
    # variable: (row_idx, pos_idx)
    # forward-check domains per row

    # Precompute per-row feasible candidate lists lazily.
    def row_candidates(row):
        # returns list of x (ascending) such that all diffs x - a for a in row
        # are positive, <= total_diffs, and unused
        last = row[-1]
        out = []
        x = last + 1
        while x - last <= total_diffs:
            ok = True
            for a in row:
                d = x - a
                if d > total_diffs or used[d]:
                    ok = False
                    break
            if ok:
                out.append(x)
            x += 1
        return out

    # We'll do a recursive search over positions in a fixed global order.
    # Order: fill row 0 fully, then row 1, etc. Within a row, positions 1..k.
    # Use forward checking: maintain for each row the set of used diffs.

    sysrecursion = 0

    def backtrack(ri, pos):
        nonlocal sysrecursion
        if time.time() >= deadline:
            return False
        if ri == n_rows:
            return True
        if pos == 0:
            rows.append([0])
            return backtrack(ri, 1)
        row = rows[ri]
        cands = row_candidates(row)
        if not cands:
            return False
        # choose candidate order
        if order == "min":
            cands = cands
        elif order == "max":
            cands = cands[::-1]
        else:
            # alternate: even positions min, odd max (heuristic)
            cands = cands if (pos % 2 == 1) else cands[::-1]

        for x in cands:
            sysrecursion += 1
            # apply
            diffs = [x - a for a in row]
            for d in diffs:
                used[d] = True
            row.append(x)
            if backtrack(ri, pos + 1):
                return True
            # undo
            row.pop()
            for d in diffs:
                used[d] = False
        return False

    ok = backtrack(0, 0)
    if ok:
        # trim trailing empty rows if any
        while rows and rows[-1] == [0]:
            rows.pop()
        if len(rows) == n_rows:
            return rows
    return None

# EVOLVE-BLOCK-END
