"""
Difference triangle set construction using simulated annealing with
structured row-level moves.
"""
import time
import random
import math


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope), subject to all positive
    differences within rows being distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Total number of differences: n_rows * C(k+1, 2)
    total_diffs = n_rows * k * (k + 1) // 2

    # Lower bound on scope: we need at least total_diffs distinct positive differences
    # from 1..m, so m >= total_diffs
    lb = total_diffs

    def get_scope(rows):
        return max(max(r) for r in rows)

    def get_diffs(rows):
        """Return set of all differences."""
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        """Check if all differences are distinct."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_construct(bias=0.0, seed=42):
        """Greedy construction with optional randomization."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    if bias > 0 and rng.random() < bias and len(row) > 1:
                        continue
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def row_diffs(row):
        """Return set of differences for a single row."""
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def rebuild_row(row_idx, rows, used, rng, scope_limit):
        """
        Try to rebuild a single row to fit within scope_limit.
        Returns (new_row, new_used_diffs) or (None, None) if failed.
        """
        best_row = None
        for attempt in range(50):
            row = [0]
            x = 0
            row_used = set()
            success = True
            while len(row) < k + 1:
                x += 1
                if x > scope_limit:
                    success = False
                    break
                new_diffs = {x - a for a in row}
                if not (new_diffs & used) and not (new_diffs & row_used):
                    # Sometimes skip to diversify
                    if rng.random() < 0.1 and len(row) > 1:
                        continue
                    row.append(x)
                    row_used |= new_diffs
            if success and len(row) == k + 1:
                if best_row is None or row[-1] < best_row[-1]:
                    best_row = row
                    if best_row[-1] <= scope_limit:
                        break
        if best_row is not None:
            return best_row, row_diffs(best_row)
        return None, None

    def simulated_annealing(initial_rows, max_iters, initial_temp, cooling_rate):
        """
        Simulated annealing to reduce scope.
        Moves: pick a row, rebuild it with a tighter scope constraint.
        """
        rng = random.Random(12345)
        rows = [list(r) for r in initial_rows]
        current_scope = get_scope(rows)
        best_rows = [list(r) for r in rows]
        best_scope = current_scope

        # Build used set
        used = set()
        for row in rows:
            used |= row_diffs(row)

        temp = initial_temp
        accepted = 0
        rejected = 0

        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break

            # Pick a row to modify
            row_idx = rng.randint(0, n_rows - 1)

            # Target scope: try to reduce the maximum
            target_scope = current_scope - rng.randint(1, max(1, temp))
            target_scope = max(target_scope, lb)

            # Remove current row's diffs from used
            old_diffs = row_diffs(rows[row_idx])
            for d in old_diffs:
                used.discard(d)

            # Try to rebuild the row
            new_row, new_diffs = rebuild_row(row_idx, rows, used, rng, target_scope)

            if new_row is not None:
                new_scope = max(get_scope(rows[:row_idx] + [new_row] + rows[row_idx+1:]), 0)
                # Actually compute properly
                all_rows = rows[:row_idx] + [new_row] + rows[row_idx+1:]
                new_scope_val = get_scope(all_rows)

                if new_scope_val <= current_scope:
                    # Accept
                    rows[row_idx] = new_row
                    used |= new_diffs
                    current_scope = new_scope_val
                    accepted += 1
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in rows]
                else:
                    # Reject
                    rows[row_idx] = list(rows[row_idx])  # restore
                    used |= old_diffs
                    rejected += 1
            else:
                # Failed to rebuild, restore
                used |= old_diffs
                rejected += 1

            temp *= cooling_rate

        return best_rows, best_scope

    # --- Main search ---

    # Step 1: Get initial solution via greedy
    initial_rows = greedy_construct(bias=0.0, seed=42)
    initial_scope = get_scope(initial_rows)

    # Try a few different greedy seeds to find a good starting point
    best_initial = initial_rows
    best_initial_scope = initial_scope
    for seed in [1, 2, 3, 7, 13, 42, 99, 123, 456, 789]:
        if time.time() > deadline - 60.0:
            break
        rows = greedy_construct(bias=0.1, seed=seed)
        s = get_scope(rows)
        if s < best_initial_scope:
            best_initial_scope = s
            best_initial = rows

    best_rows = [list(r) for r in best_initial]
    best_scope = best_initial_scope

    # Step 2: Simulated annealing with multiple restarts
    sa_configs = [
        (5000, 30, 0.99995),
        (5000, 50, 0.9999),
        (3000, 20, 0.99998),
        (4000, 40, 0.99992),
    ]

    for max_iters, init_temp, cooling in sa_configs:
        if time.time() > deadline - 20.0:
            break

        # Use different starting points
        start_rows = [list(r) for r in best_rows]
        sa_rows, sa_scope = simulated_annealing(start_rows, max_iters, init_temp, cooling)

        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows

        # Also try SA from a different greedy start
        if time.time() > deadline - 20.0:
            break
        alt_rows = greedy_construct(bias=0.2, seed=777)
        alt_scope = get_scope(alt_rows)
        if alt_scope <= best_scope + 10:
            sa_rows2, sa_scope2 = simulated_annealing(alt_rows, max_iters // 2, init_temp, cooling)
            if sa_scope2 < best_scope:
                best_scope = sa_scope2
                best_rows = sa_rows2

    # Step 3: Local improvement - try to reduce each row's max
    def local_improve(rows, max_iters=1000):
        """Try to reduce scope by shrinking individual rows."""
        rows = [list(r) for r in rows]
        current_scope = get_scope(rows)
        best_scope = current_scope
        best_rows = [list(r) for r in rows]
        rng = random.Random(42)

        used = set()
        for row in rows:
            used |= row_diffs(row)

        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break

            # Find row with largest last element
            max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            row = rows[max_idx]
            old_last = row[-1]

            if old_last <= lb:
                break

            # Try to rebuild this row with a smaller scope
            target = old_last - 1
            old_diffs = row_diffs(row)
            for d in old_diffs:
                used.discard(d)

            new_row, new_diffs = rebuild_row(max_idx, rows, used, rng, target)

            if new_row is not None:
                rows[max_idx] = new_row
                used |= new_diffs
                new_scope = get_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
            else:
                used |= old_diffs

        return best_rows, best_scope

    if time.time() < deadline - 10.0:
        improved_rows, improved_scope = local_improve(best_rows, max_iters=2000)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    # Final validation
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy_construct(bias=0.0, seed=42)

    return best_rows
