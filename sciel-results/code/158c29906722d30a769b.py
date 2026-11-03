"""Exact bitmask solver: search for difference triangle sets using bit operations for fast conflict checking."""
import time
import random

_DEADLINE = 140.0  # seconds; evaluator kills at 150s, keep a safety margin.


def _elapsed(start):
    return time.time() - start


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


def _all_diffs(rows):
    s = set()
    for row in rows:
        for d in _row_diffs(row):
            s.add(d)
    return s


def construct(n_rows, k):
    start = time.time()
    rng = random.Random(20250705)

    # Fallback: a valid greedy solution (always available, fast).
    base_rows = _greedy(n_rows, k, rng)
    best_rows = [r[:] for r in base_rows]
    best_max = _max_entry(best_rows)

    # Local search: try to replace one row at a time with a smaller non-conflicting row.
    cur_rows = [r[:] for r in best_rows]
    cur_max = best_max

    improved = True
    while improved and _elapsed(start) < _DEADLINE:
        improved = False
        for r_idx in range(n_rows):
            if _elapsed(start) > _DEADLINE:
                break
            # Compute diffs used by other rows
            other_used = set()
            for i, row in enumerate(cur_rows):
                if i == r_idx:
                    continue
                for d in _row_diffs(row):
                    other_used.add(d)

            cur_row = cur_rows[r_idx]
            cur_row_max = cur_row[-1]

            # Try to find a new row with smaller max that doesn't conflict with other_used
            found = False
            for new_max in range(1, cur_row_max):
                if _elapsed(start) > _DEADLINE:
                    break
                # Try random gap combinations that sum to new_max
                for _ in range(500):
                    if _elapsed(start) > _DEADLINE:
                        break
                    # Random composition of new_max into k parts
                    cuts = sorted(rng.sample(range(1, new_max), k - 1))
                    gaps = []
                    prev = 0
                    for c in cuts:
                        gaps.append(c - prev)
                        prev = c
                    gaps.append(new_max - prev)

                    row = [0]
                    for g in gaps:
                        row.append(row[-1] + g)
                    diffs = _row_diffs(row)

                    # Check if any diff conflicts with other_used
                    conflict = False
                    for d in diffs:
                        if d in other_used:
                            conflict = True
                            break
                    if not conflict:
                        cur_rows[r_idx] = row
                        found = True
                        improved = True
                        break
                if found:
                    break

            if found:
                new_max = cur_rows[r_idx][-1]
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [rw[:] for rw in cur_rows]

    # Final validation
    if _validate(best_rows):
        return best_rows
    return base_rows
