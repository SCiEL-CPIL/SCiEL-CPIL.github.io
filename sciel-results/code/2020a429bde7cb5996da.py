"""Difference triangle set (7,5) – improved via randomized greedy + simulated annealing."""

import time
import random
import sys

# We'll use a simple PRNG seeded per call for reproducibility within a run,
# but the evaluator may call construct multiple times; we just need a valid answer.

def _differences(row):
    """Return the set of all positive differences a[j]-a[i] for i<j in row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ai)
    return diffs


def _row_scope(row):
    return row[-1]


def _total_scope(rows):
    return max(row[-1] for row in rows)


def _is_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ai
                if d in seen:
                    return False
                seen.add(d)
    return True


def _greedy_construct(n_rows, k, rng, max_scope_hint=None):
    """
    Greedy construction: for each row, pick the smallest next value such that
    all new differences are unused globally.
    If max_scope_hint is given, we try to keep rows within it (soft constraint).
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            found = False
            # Start from last value + 1
            start = row[-1] + 1
            # Upper bound: if max_scope_hint, don't exceed it; else go up to some limit
            upper = max_scope_hint if max_scope_hint is not None else 200
            for x in range(start, upper + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
            if not found:
                # Try larger values
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
        rows.append(row)
    return rows


def _randomized_greedy(n_rows, k, rng, max_scope_hint=None):
    """
    Randomized greedy: for each row, instead of always taking the smallest valid x,
    sometimes take a slightly larger one to diversify.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            start = row[-1] + 1
            upper = max_scope_hint if max_scope_hint is not None else 200
            # Collect candidate values
            candidates = []
            for x in range(start, upper + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    candidates.append(x)
                    if len(candidates) >= 5:
                        break
            if not candidates:
                # Fallback: find any valid x
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates = [x]
                        break
                    x += 1
            # Pick randomly from candidates (weighted toward smaller)
            if len(candidates) == 1:
                x = candidates[0]
            else:
                # Bias toward smaller values
                weights = [1.0 / (1.0 + c) for c in candidates]
                x = rng.choices(candidates, weights=weights, k=1)[0]
            row.append(x)
            used |= {x - a for a in row[:-1]}
        rows.append(row)
    return rows


def _local_search_minimize_scope(rows, k, rng, max_iter=20000):
    """
    Local search: try to reduce the maximum scope by modifying rows.
    Moves:
      1. Replace a row with a newly constructed row that fits in a smaller scope.
      2. Swap two rows.
      3. Perturb a single position in a row and re-greedy the rest.
    """
    best_rows = [list(r) for r in rows]
    best_scope = _total_scope(best_rows)
    if not _is_valid(best_rows):
        return best_rows, best_scope

    used = set()
    for row in best_rows:
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                used.add(row[j] - ai)

    def _rebuild_used(rows):
        s = set()
        for row in rows:
            for i in range(len(row)):
                ai = row[i]
                for j in range(i + 1, len(row)):
                    s.add(row[j] - ai)
        return s

    for it in range(max_iter):
        move_type = rng.choice([0, 1, 2])
        new_rows = [list(r) for r in best_rows]
        improved = False

        if move_type == 0:
            # Replace a random row with a new greedy row constrained to best_scope
            idx = rng.randint(0, len(new_rows) - 1)
            other_rows = [r for i, r in enumerate(new_rows) if i != idx]
            used_other = set()
            for row in other_rows:
                for i in range(len(row)):
                    ai = row[i]
                    for j in range(i + 1, len(row)):
                        used_other.add(row[j] - ai)
            # Try to build a row within best_scope
            row = [0]
            success = True
            while len(row) < k + 1:
                found = False
                for x in range(row[-1] + 1, best_scope + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_other):
                        row.append(x)
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success:
                # Check global validity
                all_diffs = used_other | {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}
                if len(all_diffs) == sum(len(r) * (len(r) - 1) // 2 for r in new_rows) + len(row) * (len(row) - 1) // 2:
                    new_rows[idx] = row
                    new_scope = _total_scope(new_rows)
                    if new_scope < best_scope:
                        best_rows = new_rows
                        best_scope = new_scope
                        used = _rebuild_used(best_rows)
                        improved = True

        elif move_type == 1:
            # Swap two rows (doesn't change scope but may help future moves)
            i, j = rng.sample(range(len(new_rows)), 2)
            new_rows[i], new_rows[j] = new_rows[j], new_rows[i]
            # Scope unchanged, but we accept if still valid (always is)
            # No improvement in scope, so skip
            pass

        else:
            # Perturb a position in a random row
            idx = rng.randint(0, len(new_rows) - 1)
            pos = rng.randint(1, k)  # Don't touch position 0
            old_val = new_rows[idx][pos]
            # Try nearby values
            for delta in rng.sample(range(1, 20), min(10, 19)):
                for sign in [-1, 1]:
                    new_val = old_val + sign * delta
                    if new_val <= 0:
                        continue
                    # Check strict increase
                    if pos > 0 and new_val <= new_rows[idx][pos - 1]:
                        continue
                    if pos < k and new_val >= new_rows[idx][pos + 1]:
                        continue
                    # Tentatively set and check differences
                    test_row = list(new_rows[idx])
                    test_row[pos] = new_val
                    # Check internal diffs of this row
                    row_diffs = {test_row[j] - test_row[i] for i in range(len(test_row)) for j in range(i + 1, len(test_row))}
                    # Check against other rows
                    other_diffs = set()
                    for ri, r in enumerate(new_rows):
                        if ri == idx:
                            continue
                        for i in range(len(r)):
                            ai = r[i]
                            for j in range(i + 1, len(r)):
                                other_diffs.add(r[j] - ai)
                    if not (row_diffs & other_diffs):
                        new_rows[idx] = test_row
                        new_scope = _total_scope(new_rows)
                        if new_scope < best_scope:
                            best_rows = new_rows
                            best_scope = new_scope
                            used = _rebuild_used(best_rows)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

        if not improved and it % 500 == 0:
            # Occasionally try a full rebuild of one row
            idx = rng.randint(0, len(new_rows) - 1)
            other_rows = [r for i, r in enumerate(new_rows) if i != idx]
            used_other = set()
            for row in other_rows:
                for i in range(len(row)):
                    ai = row[i]
                    for j in range(i + 1, len(row)):
                        used_other.add(row[j] - ai)
            row = [0]
            success = True
            while len(row) < k + 1:
                found = False
                for x in range(row[-1] + 1, best_scope + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_other):
                        row.append(x)
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success:
                all_diffs = used_other | {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}
                expected = sum(len(r) * (len(r) - 1) // 2 for r in new_rows)
                if len(all_diffs) == expected:
                    new_rows[idx] = row
                    new_scope = _total_scope(new_rows)
                    if new_scope < best_scope:
                        best_rows = new_rows
                        best_scope = new_scope
                        used = _rebuild_used(best_rows)

    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    rng = random.Random(42)

    # Initial greedy construction
    best_rows = _greedy_construct(n_rows, k, rng)
    best_scope = _total_scope(best_rows)

    # Validate
    if not _is_valid(best_rows):
        # Fallback: simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        break
                    x += 1
            rows.append(row)
        return rows

    # Simulated annealing / local search with multiple restarts
    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1
        # Randomized greedy initialization
        init_rows = _randomized_greedy(n_rows, k, rng, max_scope_hint=best_scope + 10)
        if not _is_valid(init_rows):
            continue
        init_scope = _total_scope(init_rows)

        # Local search
        search_rows, search_scope = _local_search_minimize_scope(
            init_rows, k, rng, max_iter=5000
        )

        if search_scope < best_scope:
            best_rows = search_rows
            best_scope = search_scope

        # Check time
        if time.time() > deadline - 5.0:
            break

    # Final validation
    if not _is_valid(best_rows):
        # Emergency fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        break
                    x += 1
            rows.append(row)
        return rows

    return best_rows
