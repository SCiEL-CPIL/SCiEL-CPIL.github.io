"""Stronger solver: greedy seed + bounded randomized local search to minimize the max entry."""
import time
import random

_DEADLINE = 145.0  # seconds; evaluator kills at 150s, keep a safety margin.


def _row_diffs(row):
    """All positive differences a_j - a_i for i < j in a single strictly-increasing row."""
    d = []
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.append(row[j] - ai)
    return d


def _validate(rows):
    seen = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for a, b in zip(row, row[1:]):
            if b <= a:
                return False
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _greedy_row(used, k):
    """Build one row greedily: smallest next mark whose diffs are all unused."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        ok = True
        for a in row:
            if (x - a) in used:
                ok = False
                break
        if ok:
            row.append(x)
            for a in row[:-1]:
                used.add(x - a)
    return row


def _greedy(n_rows, k, rng):
    """Plain greedy construction (deterministic order)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(used, k))
    return rows


def _max_entry(rows):
    return max(r[-1] for r in rows)


def construct(n_rows, k):
    start = time.time()
    rng = random.Random(20250705)

    # Fallback: a valid greedy solution (always available, fast).
    base_rows = _greedy(n_rows, k, rng)
    best_rows = [r[:] for r in base_rows]
    best_max = _max_entry(best_rows)

    def _elapsed():
        return time.time() - start

    # Local search: repeatedly try to lower the max entry.
    # Two move types:
    #   1) Replace one non-first mark with a smaller unused value (shrinks a row).
    #   2) Swap two marks between rows (2-opt style) to relieve conflicts.
    # We only accept moves that keep all differences distinct and never increase the max.

    # Precompute: global used-diff set for the current best solution.
    def _diffs_of(rows):
        s = set()
        for row in rows:
            for d in _row_diffs(row):
                s.add(d)
        return s

    cur_rows = [r[:] for r in base_rows]
    cur_used = _diffs_of(cur_rows)
    cur_max = best_max

    # Candidate pool of small values to try for "lower" moves.
    # We search values in [1, cur_max].
    # For speed, precompute for each row the set of diffs that row uses, so we can
    # quickly check if a new value conflicts with that row's other marks.
    def _row_diffset(row):
        return set(_row_diffs(row))

    cur_row_used = [_row_diffset(r) for r in cur_rows]

    # A value v is usable in row r (at any position) iff for every a in row r,
    # |v - a| is not in cur_used and v is not equal to any a in row r.
    # But we must also keep the row strictly increasing and starting at 0.
    # Simpler: only allow replacing row[r][pos] with a value v such that
    #   row[r][pos-1] < v < row[r][pos+1]  (or boundaries), and all new diffs
    #   (v - a) for a in row[r], a != row[r][pos], are unused.

    # To keep it fast, we limit the number of iterations and use a deadline.
    # We also periodically recompute the global used set.

    # We'll do a simple simulated-annealing-like loop: try random (row, pos, new_val),
    # accept if valid and max doesn't increase; accept with small probability if it
    # does (to escape local minima).

    def _try_move(rows, row_used, used, r, pos, new_val):
        """Attempt to set rows[r][pos] = new_val. Return (new_rows, new_row_used, new_used, new_max) or None."""
        row = rows[r]
        # Bounds: keep strictly increasing.
        lo = row[pos - 1] if pos > 0 else 0
        hi = row[pos + 1] if pos < len(row) - 1 else 10**9
        if new_val <= lo or new_val >= hi:
            return None
        if new_val == row[pos]:
            return None
        # Check conflicts with other marks in this row.
        old_val = row[pos]
        # Diffs that will be removed: old_val - a and a - old_val for a in row, a != old_val.
        # Diffs that will be added: new_val - a and a - new_val for a in row, a != new_val.
        # Since row is increasing, for a < old_val: diff = old_val - a; for a > old_val: diff = a - old_val.
        # We need all added diffs to be unused (except those that are also removed, i.e., coincident).
        added = set()
        removed = set()
        for i, a in enumerate(row):
            if i == pos:
                continue
            if a < old_val:
                removed.add(old_val - a)
            else:
                removed.add(a - old_val)
            if a < new_val:
                added.add(new_val - a)
            else:
                added.add(a - new_val)
        # The net new diffs are added - removed; these must not be in (used - removed).
        # Equivalently: for d in added: if d not in removed and d in used: conflict.
        for d in added:
            if d not in removed and d in used:
                return None
        # Also, new_val must not equal any other a in row (guaranteed by strict increase check).
        # Build new row.
        new_row = row[:]
        new_row[pos] = new_val
        new_rows = rows[:]
        new_rows[r] = new_row
        # Update used set.
        new_used = used - removed
        new_used |= added
        # Update row_used.
        new_row_used = row_used[:]
        new_row_used[r] = new_used & _row_diffset(new_row) if False else set(_row_diffs(new_row))
        # Actually recompute this row's diff set.
        new_row_used[r] = set(_row_diffs(new_row))
        new_max = max(rw[-1] for rw in new_rows)
        return new_rows, new_row_used, new_used, new_max

    # Main loop.
    # We cap the number of iterations to stay within budget.
    max_iters = 200000
    iters = 0
    T = float(cur_max)  # temperature for occasional uphill moves.
    T_min = 0.1
    cooling = 0.9999

    while iters < max_iters:
        if _elapsed() > _DEADLINE:
            break
        iters += 1

        # Pick a random row and position.
        r = rng.randrange(n_rows)
        row = cur_rows[r]
        # Positions 1..k (not 0, which is fixed at 0).
        pos = rng.randrange(1, len(row))

        # Choose a candidate new_val in a range around the current value.
        cur_val = row[pos]
        # Bias toward smaller values to reduce the max.
        # Sample new_val from [1, cur_val - 1] if cur_val > 1, else skip.
        if cur_val <= 1:
            continue
        # With some probability, sample from a wider range to allow restructuring.
        if rng.random() < 0.8:
            new_val = rng.randrange(1, cur_val)
        else:
            # Sample from a broader range, possibly larger, to allow swaps.
            new_val = rng.randrange(1, min(cur_max + 10, 200))

        res = _try_move(cur_rows, cur_row_used, cur_used, r, pos, new_val)
        if res is None:
            continue
        new_rows, new_row_used, new_used, new_max = res

        # Acceptance: always accept if new_max <= cur_max.
        # If new_max > cur_max, accept with probability exp(-(new_max - cur_max)/T).
        if new_max <= cur_max:
            cur_rows, cur_row_used, cur_used, cur_max = new_rows, new_row_used, new_used, new_max
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [rw[:] for rw in cur_rows]
        else:
            # Uphill move: only if it might help later; keep probability low.
            delta = new_max - cur_max
            if delta <= 5 and rng.random() < (0.01 / (1.0 + delta)):
                cur_rows, cur_row_used, cur_used, cur_max = new_rows, new_row_used, new_used, new_max

        # Cool down.
        T = max(T_min, T * cooling)

    # Final: return the best valid solution found.
    if _validate(best_rows):
        return best_rows
    # Fallback to the greedy solution (always valid).
    return base_rows
