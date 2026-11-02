"""Difference triangle set (7,5): minimize the largest entry over rows of k+1 increasing ints from 0,
with all within-row positive differences globally distinct. Uses greedy init + simulated-annealing local search."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _row_diffs(row):
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d

def _max_entry(rows):
    return max(row[-1] for row in rows)

def construct(n_rows, k):
    deadline = time.time() + 140.0
    rng = random.Random(12345)
    n_marks = k + 1

    # ---- Greedy initial solution ----
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < n_marks:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)

    best = [r[:] for r in rows]
    best_val = _max_entry(best)

    # Maintain global used set incrementally
    for r in rows:
        used |= _row_diffs(r)

    def apply_move(row_idx, pos, new_val):
        """Set rows[row_idx][pos]=new_val if valid. Returns (old_val, old_diffs, new_diffs) or None."""
        row = rows[row_idx]
        old_val = row[pos]
        if old_val == new_val:
            return None
        if pos > 0 and new_val <= row[pos - 1]:
            return None
        if pos < n_marks - 1 and new_val >= row[pos + 1]:
            return None
        new_row = row[:]
        new_row[pos] = new_val
        old_diffs = _row_diffs(row)
        new_diffs = _row_diffs(new_row)
        other_used = used - old_diffs
        if (new_diffs - old_diffs) & other_used:
            return None
        # commit
        used -= old_diffs
        used |= new_diffs
        row[:] = new_row
        return (old_val, old_diffs, new_diffs)

    def undo_move(row_idx, pos, old_val, old_diffs, new_diffs):
        used -= new_diffs
        used |= old_diffs
        rows[row_idx][pos] = old_val

    cur_val = best_val
    T = 4.0
    T_min = 0.03
    cooling = 0.9998

    def exp_neg(x):
        if x > 50:
            return 0.0
        return math.exp(-x)

    while time.time() < deadline:
        if rng.random() < 0.5:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randrange(n_rows)

        if rng.random() < 0.6:
            pos = n_marks - 1
        else:
            pos = rng.randint(1, n_marks - 2) if n_marks - 2 >= 1 else 1

        row = rows[row_idx]
        old_val = row[pos]
        span = max(1, int(0.2 * (old_val + 1)))
        if rng.random() < 0.7:
            lo = (row[pos - 1] + 1) if pos > 0 else 1
            hi = old_val - 1
            lo = max(lo, old_val - span)
            if hi < lo:
                continue
            new_val = rng.randint(lo, hi)
        else:
            lo = old_val + 1
            hi = (row[pos + 1] - 1) if pos < n_marks - 1 else old_val + span
            hi = min(hi, old_val + span)
            if hi < lo:
                continue
            new_val = rng.randint(lo, hi)

        res = apply_move(row_idx, pos, new_val)
        if res is None:
            continue
        old_val, old_diffs, new_diffs = res
        new_scope = _max_entry(rows)
        delta = new_scope - cur_val

        if delta <= 0:
            cur_val = new_scope
            if cur_val < best_val:
                best_val = cur_val
                best = [r[:] for r in rows]
        else:
            if T > T_min and rng.random() < exp_neg(delta / T):
                cur_val = new_scope
            else:
                undo_move(row_idx, pos, old_val, old_diffs, new_diffs)

        T = max(T_min, T * cooling)

    return best

# EVOLVE-BLOCK-END
