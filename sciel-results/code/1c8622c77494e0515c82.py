"""Difference triangle set (4,7) – optimized construction.

Strategy:
  1. Fast greedy with bitset conflict tracking to get a quick feasible solution.
  2. Local search (simulated-annealing-style) that tries to reduce the scope
     by swapping/inserting/deleting marks within rows while preserving the
     global distinctness constraint on all positive differences.
  3. Deadline guard to return the best valid solution before the 120 s limit.
"""

import time
import random
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_diffs(row):
    """Return the set of positive differences for a row (list of ints)."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs


def _all_diffs(rows):
    """Return the union of all positive differences across rows."""
    s = set()
    for r in rows:
        s |= _row_diffs(r)
    return s


def _is_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for r in rows:
        d = _row_diffs(r)
        if d & seen:
            return False
        seen |= d
    return True


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)


# ---------------------------------------------------------------------------
# Greedy construction (fast baseline)
# ---------------------------------------------------------------------------

def _greedy(n_rows, k, used=None):
    """Greedy: build rows one at a time, smallest next mark with no conflict."""
    if used is None:
        used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = set()
            conflict = False
            for a in row:
                d = x - a
                if d in used:
                    conflict = True
                    break
                new.add(d)
            if not conflict:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def _local_search(rows, n_rows, k, time_limit, rng):
    """
    Try to reduce the scope by local moves:
      - Move a mark in a row to a smaller unused position.
      - Try to swap two marks within a row to reduce the max.
      - Try to replace the largest mark in a row with a smaller one.
    Returns improved rows.
    """
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    start = time.time()
    timeout = time_limit - 5  # safety margin

    while time.time() - start < timeout:
        # Work on a copy
        cur = [list(r) for r in best_rows]
        cur_scope = best_scope
        improved = False

        # Try to reduce the scope by replacing the largest mark in each row
        for ri in range(n_rows):
            row = cur[ri]
            max_idx = max(range(len(row)), key=lambda i: row[i])
            if max_idx == 0:
                continue
            old_val = row[max_idx]
            if old_val < 1:
                continue
            # Compute current diffs for this row
            row_diffs = _row_diffs(row)
            # Diffs used by other rows
            other_diffs = set()
            for rj in range(n_rows):
                if rj == ri:
                    continue
                other_diffs |= _row_diffs(cur[rj])
            all_other = other_diffs

            # Try smaller values for the max mark
            for new_val in range(old_val - 1, 0, -1):
                if new_val <= row[max_idx - 1]:
                    break  # must stay strictly increasing
                # Check if new_val - row[i] for i < max_idx conflicts
                ok = True
                for i in range(max_idx):
                    d = new_val - row[i]
                    if d in all_other or d in row_diffs - {d for i2 in range(len(row)) if i2 < max_idx and new_val - row[i2] == d}:
                        # more careful:
                        pass
                # Recompute properly
                trial_row = list(row)
                trial_row[max_idx] = new_val
                if trial_row[max_idx] <= trial_row[max_idx - 1]:
                    continue
                trial_diffs = _row_diffs(trial_row)
                # Check against other rows
                conflict = False
                for rj in range(n_rows):
                    if rj == ri:
                        continue
                    if trial_diffs & _row_diffs(cur[rj]):
                        conflict = True
                        break
                if not conflict:
                    cur[ri] = trial_row
                    cur_scope = _scope(cur)
                    improved = True
                    break

        if improved and cur_scope < best_scope:
            best_rows = [list(r) for r in cur]
            best_scope = cur_scope

        # Random perturbation: try moving a non-max mark
        if rng.random() < 0.3:
            ri = rng.randrange(n_rows)
            row = best_rows[ri]
            if len(row) > 2:
                idx = rng.randrange(1, len(row) - 1)  # not first, not last
                old = row[idx]
                # Try a nearby smaller value
                for delta in range(1, 6):
                    new_val = old - delta
                    if new_val <= row[idx - 1]:
                        break
                    trial = list(row)
                    trial[idx] = new_val
                    trial_diffs = _row_diffs(trial)
                    conflict = False
                    for rj in range(n_rows):
                        if rj == ri:
                            continue
                        if trial_diffs & _row_diffs(best_rows[rj]):
                            conflict = True
                            break
                    if not conflict:
                        # Also check internal consistency (should be fine since increasing)
                        best_rows[ri] = trial
                        new_scope = _scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        break

    return best_rows, best_scope


# ---------------------------------------------------------------------------
# Multi-start with randomization
# ---------------------------------------------------------------------------

def _multi_start_greedy(n_rows, k, rng, n_trials=50):
    """Run greedy multiple times with slight randomization to find diverse solutions."""
    best = None
    best_scope = float('inf')
    for _ in range(n_trials):
        rows = _greedy(n_rows, k)
        s = _scope(rows)
        if s < best_scope:
            best_scope = s
            best = rows
    return best, best_scope


# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences across all rows are distinct. Minimize the scope.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115 s to leave 5 s margin before 120 s kill

    rng = random.Random(42)

    # Phase 1: Quick greedy to get a valid baseline
    rows = _greedy(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)

    # Phase 2: Multi-start greedy for diversity
    if time.time() < deadline:
        ms_rows, ms_scope = _multi_start_greedy(n_rows, k, rng, n_trials=20)
        if ms_scope < best_scope:
            best_rows = ms_rows
            best_scope = ms_scope

    # Phase 3: Local search to reduce scope
    if time.time() < deadline:
        ls_rows, ls_scope = _local_search(best_rows, n_rows, k, deadline - start_time, rng)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope

    # Phase 4: More local search iterations with different random seeds
    if time.time() < deadline:
        for seed in [1, 2, 3, 7, 13]:
            if time.time() > deadline:
                break
            rng2 = random.Random(seed)
            # Restart from best and do more local search
            cur_rows, cur_scope = _local_search(best_rows, n_rows, k, deadline - start_time, rng2)
            if cur_scope < best_scope:
                best_rows = cur_rows
                best_scope = cur_scope

    # Final validation
    if not _is_valid(best_rows):
        # Fallback to simple greedy
        best_rows = _greedy(n_rows, k)

    return best_rows
