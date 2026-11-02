"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _greedy_row(used, k):
    """Build one row greedily given the set of already-used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used.update(new_diffs)
    return row

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return the union of all differences across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _try_compact_row(row, other_diffs, k):
    """Try to compact a single row: pull each element left to smallest valid position.
    Returns (new_row, success)."""
    new_row = [0]
    for pos in range(1, len(row)):
        orig = row[pos]
        # Try to place at smallest x >= new_row[-1]+1
        x = new_row[-1] + 1
        placed = False
        while x <= orig:
            new_diffs = {x - a for a in new_row}
            if not new_diffs & other_diffs:
                new_row.append(x)
                placed = True
                break
            x += 1
        if not placed:
            return row, False
    return new_row, True

def _rebuild_row_greedy(used_without_row, k, start_hint=0):
    """Rebuild a row greedily given differences used by all OTHER rows.
    Returns (row, diffs_in_row)."""
    row = [0]
    x = start_hint
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used_without_row:
            row.append(x)
            used_without_row.update(new_diffs)
    return row

def construct(n_rows, k):
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(42)

    # Phase 1: Greedy initialization
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(used, k))
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]

    # Phase 2: Simulated Annealing
    # State: rows, used_diffs
    cur_rows = [list(r) for r in rows]
    cur_used = _all_diffs(cur_rows)
    cur_scope = _scope(cur_rows)

    T = 50.0  # initial temperature
    T_min = 0.1
    cooling = 0.995

    while time.time() < deadline and T > T_min:
        # Pick a random row to perturb
        ri = rng.randrange(n_rows)
        row = cur_rows[ri]

        # Compute differences used by all other rows
        other_diffs = set()
        for idx, r in enumerate(cur_rows):
            if idx == ri:
                continue
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    other_diffs.add(r[j] - r[i])

        # Try to compact this row
        new_row, ok = _try_compact_row(row, other_diffs, k)
        if ok and new_row != row:
            new_scope = max(_scope(cur_rows[:ri] + [new_row] + cur_rows[ri+1:]), 0)
            # Check all diffs are still distinct
            all_d = set(other_diffs)
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    all_d.add(new_row[j] - new_row[i])
            # Count total diffs: should be n_rows * k*(k+1)/2
            expected = n_rows * k * (k + 1) // 2
            if len(all_d) == expected:
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < min(1.0, (-delta / T) if delta > 0 else 1.0):
                    # Accept
                    cur_rows[ri] = new_row
                    cur_scope = new_scope
                    cur_used = _all_diffs(cur_rows)
                    if cur_scope < best_scope:
                        best_scope = cur_scope
                        best_rows = [list(r) for r in cur_rows]
        else:
            # Try a random perturbation: remove last element, reinsert elsewhere
            if len(row) > 1:
                last = row[-1]
                reduced = row[:-1]
                # Try to reinsert last at a random earlier position or smaller value
                # Try several candidate positions
                best_new_row = None
                best_new_scope = cur_scope
                for _attempt in range(5):
                    # Try shifting the last element to a smaller value
                    lo = reduced[-1] + 1
                    hi = last - 1 if _attempt < 3 else last + rng.randint(0, 5)
                    if hi < lo:
                        continue
                    x = rng.randint(lo, hi)
                    cand = reduced + [x]
                    # Check validity
                    cand_diffs = set()
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            d = cand[j] - cand[i]
                            if d in other_diffs:
                                break
                            cand_diffs.add(d)
                    else:
                        # All diffs valid and not in other_diffs
                        new_scope = max(max(r) for r in cur_rows[:ri] + [cand] + cur_rows[ri+1:])
                        if new_scope < best_new_scope:
                            best_new_scope = new_scope
                            best_new_row = cand

                if best_new_row is not None:
                    delta = best_new_scope - cur_scope
                    if delta <= 0 or rng.random() < min(1.0, (-delta / T) if delta > 0 else 1.0):
                        cur_rows[ri] = best_new_row
                        cur_scope = best_new_scope
                        cur_used = _all_diffs(cur_rows)
                        if cur_scope < best_scope:
                            best_scope = cur_scope
                            best_rows = [list(r) for r in cur_rows]

        T *= cooling

    # Phase 3: Final compaction pass on best solution
    for pass_num in range(3):
        if time.time() > deadline - 5:
            break
        for ri in range(n_rows):
            other_diffs = set()
            for idx, r in enumerate(best_rows):
                if idx == ri:
                    continue
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        other_diffs.add(r[j] - r[i])
            new_row, ok = _try_compact_row(best_rows[ri], other_diffs, k)
            if ok and new_row != best_rows[ri]:
                # Verify global validity
                all_d = set(other_diffs)
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        all_d.add(new_row[j] - new_row[i])
                expected = n_rows * k * (k + 1) // 2
                if len(all_d) == expected:
                    best_rows[ri] = new_row
                    best_scope = _scope(best_rows)

    return best_rows
# EVOLVE-BLOCK-END
