"""Difference triangle set (4,7) – improved search."""
import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, subject to all positive differences
    within rows being distinct across all rows.
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit
    rng = random.Random(42)

    total_diffs = n_rows * k * (k + 1) // 2
    # Lower bound on scope: need at least total_diffs distinct positive differences,
    # max difference is scope, so scope*(scope+1)/2 >= total_diffs
    # scope >= ceil((-1 + sqrt(1 + 8*total_diffs)) / 2)
    lb = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)

    # Helper: compute all positive differences in a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def valid_rows(rows):
        """Check that all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            for d in row_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True

    def scope_of(rows):
        return max(max(r) for r in rows)

    # Greedy construction with randomization
    def randomized_greedy(rng, n_rows, k, bias=0.0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try next candidates with some randomization
                candidates = []
                for dx in range(1, 50):
                    cand = x + dx
                    new = {cand - a for a in row}
                    if not (new & used):
                        candidates.append((cand, new))
                    if len(candidates) > 20:
                        break
                if not candidates:
                    # Fallback: try larger jumps
                    for dx in range(50, 200):
                        cand = x + dx
                        new = {cand - a for a in row}
                        if not (new & used):
                            candidates.append((cand, new))
                            break
                if not candidates:
                    return None
                # Choose: mostly smallest, sometimes random
                if bias > 0 and rng.random() < bias and len(candidates) > 1:
                    idx = rng.randint(1, len(candidates) - 1)
                else:
                    idx = 0
                cand, new = candidates[idx]
                row.append(cand)
                used |= new
                x = cand
            rows.append(row)
        return rows

    # Local search: try to reduce scope by moving individual entries
    def local_search(rows, max_iters=5000):
        """Try single-entry moves to reduce the maximum entry."""
        rows = [list(r) for r in rows]
        best_scope = scope_of(rows)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # For each row, try moving each non-zero entry to a smaller value
            for ri in range(len(rows)):
                row = rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    orig = row[pos]
                    # Try reducing this entry
                    for new_val in range(orig - 1, row[pos - 1], -1):
                        row[pos] = new_val
                        # Check validity
                        all_diffs = set()
                        valid = True
                        for r in rows:
                            for d in row_diffs(r):
                                if d in all_diffs:
                                    valid = False
                                    break
                                all_diffs.add(d)
                            if not valid:
                                break
                        if valid:
                            new_scope = scope_of(rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                improved = True
                                break
                            else:
                                row[pos] = orig
                                break
                        else:
                            row[pos] = orig
                            # Don't break; try even smaller
        return rows, best_scope

    # Try pure greedy first
    best_rows = randomized_greedy(rng, n_rows, k, bias=0.0)
    if best_rows is None:
        return [[0] + list(range(1, k + 1))] * n_rows  # fallback (invalid but safe)
    best_scope = scope_of(best_rows)

    # Try multiple randomized starts and local search
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        bias = rng.uniform(0.1, 0.5)
        rows = randomized_greedy(rng, n_rows, k, bias=bias)
        if rows is None:
            continue
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s
        # Local search on this candidate
        if time.time() > deadline - 30:
            break
        improved_rows, imp_scope = local_search(rows, max_iters=2000)
        if imp_scope < best_scope:
            best_rows = improved_rows
            best_scope = imp_scope

    # If we still have time, do more local search on the best
    if time.time() < deadline - 20:
        improved_rows, imp_scope = local_search(best_rows, max_iters=5000)
        if imp_scope < best_scope:
            best_rows = improved_rows
            best_scope = imp_scope

    return best_rows
