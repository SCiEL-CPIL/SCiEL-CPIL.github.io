"""
Difference triangle set (7,5): choose 7 rows of 6 increasing integers starting at 0
so that all pairwise differences within each row are distinct across all rows,
minimizing the maximum entry.

Strategy: iterative local search (simulated-annealing style) with a global
differences-used table. Start from the greedy construction, then try random
row modifications and accept improvements (or occasionally worse moves to
escape local optima). Track the best solution found and enforce a wall-clock
deadline.
"""

import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    # Greedy initial solution
    rows = _greedy(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_score = _max_entry(best_rows)
    best_diffs = _compute_diffs(best_rows)

    # Simulated annealing / iterative improvement
    rows = [r[:] for r in best_rows]
    diffs = best_diffs
    score = best_score
    T = 50.0  # initial "temperature"
    cooling = 0.995

    while time.time() < deadline:
        # Pick a random row to modify
        r_idx = random.randrange(n_rows)
        old_row = rows[r_idx]

        # Remove this row's differences
        row_diffs = _row_diffs(old_row)
        for d in row_diffs:
            diffs[d] -= 1
        # Remove zero-diff entries? No, only positive diffs are tracked.

        # Try to rebuild this row greedily but with some randomness
        new_row = _greedy_row(k, diffs, randomize=True, max_entry=score + 20)
        if new_row is None:
            # Restore
            for d in row_diffs:
                diffs[d] += 1
            continue

        new_row_diffs = _row_diffs(new_row)
        new_max = new_row[-1]

        # Check if new row's diffs are all available
        valid = True
        for d in new_row_diffs:
            if diffs[d] >= 1:
                valid = False
                break
        if not valid:
            # Restore
            for d in row_diffs:
                diffs[d] += 1
            continue

        # Apply the change
        for d in new_row_diffs:
            diffs[d] += 1
        rows[r_idx] = new_row
        new_score = _max_entry(rows)

        if new_score < score:
            # Accept improvement
            score = new_score
            if score < best_score:
                best_score = score
                best_rows = [r[:] for r in rows]
                best_diffs = _compute_diffs(best_rows)
        else:
            # Accept worse with some probability (annealing)
            delta = new_score - score
            if delta > 0 and random.random() < (1.0 / (1.0 + delta / max(T, 0.1))):
                score = new_score
            else:
                # Revert
                for d in new_row_diffs:
                    diffs[d] -= 1
                rows[r_idx] = old_row[:]
                for d in row_diffs:
                    diffs[d] += 1
                continue

        T *= cooling

    # One final greedy improvement pass on best
    final_rows = _try_improve(best_rows, n_rows, k, deadline)
    final_score = _max_entry(final_rows)
    if final_score < best_score:
        best_rows = final_rows

    return best_rows


def _greedy(n_rows, k):
    """Greedy construction: each row takes smallest next mark with unused diffs."""
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


def _greedy_row(k, diffs, randomize=False, max_entry=200):
    """Build one row greedily given the global diffs table. Returns row or None."""
    row = [0]
    x = 0
    attempts = 0
    max_attempts = max_entry + 50
    while len(row) < k + 1 and x < max_entry:
        x += 1
        new_diffs = {x - a for a in row}
        # Check all diffs are available
        ok = True
        for d in new_diffs:
            if diffs.get(d, 0) >= 1:
                ok = False
                break
        if ok:
            row.append(x)
            attempts = 0
        else:
            if randomize and attempts > 5 and random.random() < 0.1:
                # Sometimes skip ahead
                x += random.randint(1, 3)
            attempts += 1
    if len(row) < k + 1:
        return None
    return row


def _row_diffs(row):
    """All positive pairwise differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _compute_diffs(rows):
    """Compute the global diffs table (count of how many rows use each diff)."""
    diffs = {}
    for row in rows:
        for d in _row_diffs(row):
            diffs[d] = diffs.get(d, 0) + 1
    return diffs


def _max_entry(rows):
    return max(r[-1] for r in rows)


def _try_improve(rows, n_rows, k, deadline):
    """Try a few quick greedy rebuilds of individual rows to see if we can reduce max."""
    improved = [r[:] for r in rows]
    diffs = _compute_diffs(improved)
    cur_score = _max_entry(improved)

    for _ in range(20):  # limited passes
        if time.time() > deadline:
            break
        best_improvement = None
        for r_idx in range(n_rows):
            old_row = improved[r_idx]
            row_diffs = _row_diffs(old_row)
            for d in row_diffs:
                diffs[d] -= 1

            new_row = _greedy_row(k, diffs, randomize=False, max_entry=cur_score - 1)
            if new_row is not None:
                new_diffs = _row_diffs(new_row)
                valid = all(diffs.get(d, 0) < 1 for d in new_diffs)
                if valid:
                    new_score = max(max(r[-1] for r in improved if r is not old_row), new_row[-1])
                    if new_score < cur_score:
                        for d in new_diffs:
                            diffs[d] += 1
                        improved[r_idx] = new_row
                        cur_score = new_score
                        best_improvement = True
                        break
            # Revert
            for d in row_diffs:
                diffs[d] += 1
        if best_improvement is None:
            break

    return improved
