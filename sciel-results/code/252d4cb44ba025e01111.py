"""Rewritten program for the difference triangle set task.
Uses randomized greedy construction with simulated-annealing-style row replacement,
plus a deterministic greedy baseline, all under an explicit wall-clock deadline."""

import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences a_j - a_i (i<j) within a row are distinct across all rows.
    Minimizes the largest entry (the scope)."""
    deadline = time.time() + 140.0  # safety margin under 150s kill

    # --- Deterministic greedy baseline (fast, always valid) ---
    def greedy_baseline():
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

    best_rows = greedy_baseline()
    best_scope = max(max(r) for r in best_rows)

    # Helper: compute all positive differences in a row
    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    # Helper: check if a candidate row is valid given the set of used diffs from other rows
    def is_row_valid(candidate, other_used):
        cd = row_diffs(candidate)
        return len(cd) == len(other_used | cd)  # no overlap with other_used, and no internal dupes

    # Helper: greedy construction of a single row given a set of forbidden diffs
    def greedy_row(forbidden, max_val):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val:
                return None
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden):
                row.append(x)
        return row

    # Helper: randomized greedy row construction
    def randomized_row(forbidden, max_val, temperature=1.0):
        """Build a row greedily but with some randomization to explore alternatives."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val:
                return None
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden):
                # Accept with probability depending on temperature and how "tight" this choice is
                # Simpler: accept with some probability, otherwise skip
                if random.random() < 0.85 or temperature < 0.5:
                    row.append(x)
                # else: skip this x, try next
        if len(row) == k + 1:
            return row
        return None

    # Simulated annealing with row replacement
    # We try to improve the scope by replacing rows one at a time with better (smaller max) rows.
    current_rows = list(best_rows)
    current_scope = best_scope

    # Compute initial used set
    def compute_used(rows):
        used = set()
        for r in rows:
            used |= row_diffs(r)
        return used

    current_used = compute_used(current_rows)

    # Annealing parameters
    T = float(current_scope)  # initial temperature
    T_min = 1.0
    cooling = 0.995
    max_iter = 20000

    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break

        # Pick a random row to replace
        idx = random.randint(0, n_rows - 1)

        # Compute forbidden diffs (all diffs from other rows)
        other_used = current_used - row_diffs(current_rows[idx])

        # Try to find a row with smaller max than current_rows[idx]'s max
        old_max = max(current_rows[idx])
        # Target: try to keep max <= current_scope, ideally smaller
        target_max = current_scope

        # Attempt randomized greedy
        success = False
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
            new_row = randomized_row(other_used, target_max, temperature=T / max(T_min, 1))
            if new_row is not None:
                new_max = max(new_row)
                new_scope = max(max(r) for r in current_rows if r is not current_rows[idx]) if n_rows > 1 else 0
                new_scope = max(new_scope, new_max)
                if new_scope < current_scope:
                    # Accept improvement
                    current_rows[idx] = new_row
                    current_used = compute_used(current_rows)
                    current_scope = new_scope
                    best_rows = list(current_rows)
                    best_scope = current_scope
                    success = True
                    T = max(T * cooling, T_min)
                    break
                elif new_max < old_max and random.random() < min(1.0, T / max(new_scope - current_scope + 1, 1)):
                    # Accept if it reduces this row's max (even if overall scope doesn't change yet)
                    current_rows[idx] = new_row
                    current_used = compute_used(current_rows)
                    T = max(T * cooling, T_min)
                    success = True
                    break

        if not success:
            T = max(T * cooling, T_min)

    # Final deterministic polish: try to reduce scope by re-greedy-ing each row in order
    # Try to tighten: rebuild rows greedily but with the constraint that max <= best_scope - 1
    if time.time() < deadline - 3.0:
        target = best_scope - 1
        used = set()
        polished = []
        ok = True
        for i in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if x > target:
                    ok = False
                    break
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
            if not ok:
                break
            used |= row_diffs(row)
            polished.append(row)
        if ok:
            best_rows = polished
            best_scope = max(max(r) for r in best_rows)

    return best_rows
# EVOLVE-BLOCK-END
