"""Rewritten program for difference triangle set (7,5).
Strategy: interleaved greedy initialization + aggressive local search (HC/SA) with
last-element bias and a final polish pass. Includes a deadline guard."""

import time
import random
import copy

# Deadline: evaluator kills at 150s; use 145s as hard limit to be safe.
_HARD_DEADLINE = 145.0


def _is_valid(rows):
    """Check that all positive differences within rows are distinct across all rows."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _scope(rows):
    """Return the largest entry (max of last elements)."""
    return max(row[-1] for row in rows)


def _interleaved_greedy(n_rows, k):
    """Build rows in an interleaved fashion to balance difference usage."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    # We'll extend rows round-robin: each round, try to add one more element to each row.
    target_len = k + 1
    while any(len(r) < target_len for r in rows):
        for idx in range(n_rows):
            row = rows[idx]
            if len(row) >= target_len:
                continue
            # Try smallest next value
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
                x += 1
    return rows


def _all_diffs(rows):
    """Return set of all positive differences in rows."""
    s = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
    return s


def _try_move(rows, n_rows, k, rng, bias_last=True):
    """Try a local move: pick a random row, a random position, and try to shift it.
    Returns (new_rows, improvement) or (None, 0)."""
    row_idx = rng.randint(0, n_rows - 1)
    row = rows[row_idx]
    if len(row) < 2:
        return None, 0

    # Pick a position to modify (not position 0)
    pos = rng.randint(1, len(row) - 1)

    # Determine a range of candidate values for row[pos]
    # Must be > row[pos-1] and < row[pos+1] if pos < len(row)-1
    lo = row[pos - 1] + 1
    hi = row[pos + 1] - 1 if pos < len(row) - 1 else row[pos] + 50

    if hi <= lo:
        return None, 0

    # Try a few random candidates
    candidates = []
    if hi - lo <= 30:
        candidates = list(range(lo, hi + 1))
        rng.shuffle(candidates)
    else:
        for _ in range(20):
            c = rng.randint(lo, hi)
            candidates.append(c)

    # Compute current differences for this row
    current_diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            current_diffs.add(row[j] - row[i])

    for new_val in candidates:
        # Build new row
        new_row = row[:]
        new_row[pos] = new_val
        # Check strictly increasing
        if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
            continue

        # Compute new differences for this row
        new_diffs = set()
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                new_diffs.add(new_row[j] - new_row[i])

        # Check no collision with other rows' differences
        other_diffs = set()
        for r_idx in range(n_rows):
            if r_idx == row_idx:
                continue
            r = rows[r_idx]
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    other_diffs.add(r[j] - r[i])

        if new_diffs & other_diffs:
            continue

        # Valid move. Compute improvement (negative scope change is good)
        old_scope = _scope(rows)
        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)
        improvement = old_scope - new_scope  # positive = better

        if improvement > 0 or rng.random() < 0.3:
            return new_rows, improvement

    return None, 0


def _optimize(rows, n_rows, k, time_limit, rng):
    """Local search: hill climbing + simulated annealing."""
    current = [r[:] for r in rows]
    current_scope = _scope(current)
    best = [r[:] for r in current]
    best_scope = current_scope
    start_time = time.time()

    # Hill climbing phase
    hc_stagnation = 0
    while time.time() - start_time < time_limit * 0.4:
        new_rows, imp = _try_move(current, n_rows, k, rng)
        if new_rows is not None and imp > 0:
            current = new_rows
            current_scope = _scope(current)
            if current_scope < best_scope:
                best = [r[:] for r in current]
                best_scope = current_scope
            hc_stagnation = 0
        else:
            hc_stagnation += 1
            if hc_stagnation > 500:
                break

    # Simulated annealing phase
    sa_start = time.time()
    sa_end = start_time + time_limit * 0.85
    T = 10.0
    T_min = 0.1
    cooling = 0.9995

    while time.time() < sa_end:
        new_rows, imp = _try_move(current, n_rows, k, rng)
        if new_rows is None:
            continue
        new_scope = _scope(new_rows)
        delta = new_scope - current_scope  # negative = better
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(T, 1e-9)):
            current = new_rows
            current_scope = new_scope
            if current_scope < best_scope:
                best = [r[:] for r in current]
                best_scope = current_scope
        T = max(T * cooling, T_min)

    # Final polish: deterministic hill climbing with tighter criteria
    polish_start = time.time()
    polish_end = start_time + time_limit * 0.97
    while time.time() < polish_end:
        improved = False
        for row_idx in range(n_rows):
            row = current[row_idx]
            if len(row) < 2:
                continue
            for pos in range(1, len(row)):
                lo = row[pos - 1] + 1
                hi = row[pos + 1] - 1 if pos < len(row) - 1 else row[pos] + 20
                for new_val in range(lo, hi + 1):
                    new_row = row[:]
                    new_row[pos] = new_val
                    if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                        continue
                    new_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            new_diffs.add(new_row[j] - new_row[i])
                    other_diffs = set()
                    for r_idx in range(n_rows):
                        if r_idx == row_idx:
                            continue
                        r = current[r_idx]
                        for i in range(len(r)):
                            for j in range(i + 1, len(r)):
                                other_diffs.add(r[j] - r[i])
                    if new_diffs & other_diffs:
                        continue
                    new_scope = _scope([current[r_idx] if r_idx != row_idx else new_row for r_idx in range(n_rows)])
                    if new_scope < current_scope:
                        current[row_idx] = new_row
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best = [r[:] for r in current]
                            best_scope = current_scope
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best, best_scope


def construct(n_rows, k):
    start_time = time.time()
    rng = random.Random(42)

    # Phase 1: Interleaved greedy initialization
    rows = _interleaved_greedy(n_rows, k)
    current_scope = _scope(rows)

    # Phase 2: Local search with time budget
    time_remaining = _HARD_DEADLINE - (time.time() - start_time)
    if time_remaining > 2.0:
        rows, current_scope = _optimize(rows, n_rows, k, time_remaining - 1.0, rng)

    # Final validation
    if not _is_valid(rows):
        # Fallback to a valid greedy construction
        rows = _interleaved_greedy(n_rows, k)

    return rows
