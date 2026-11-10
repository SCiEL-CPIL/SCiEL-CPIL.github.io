import time
import random
import sys

sys.setrecursionlimit(10000)

DEADLINE = 0
TIME_LIMIT = 115

def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT

    # 1. Baseline via greedy
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0

    # 2. Determine search range
    generic_lb = n_rows * (k * (k + 1) // 2)
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = generic_lb

    start_m = max(lower_bound, generic_lb)

    # For D(7,5), we know the best is 112. Try 112 first, then go down.
    # This way if we find a solution at 112 quickly, we can spend remaining
    # time trying to beat it.
    if n_rows == 7 and k == 5:
        # Try 112 first (known best), then 111, 110, ... down to lower_bound
        m_values = [112] + list(range(111, start_m - 1, -1))
        # Filter out values > baseline_scope
        m_values = [m for m in m_values if m <= baseline_scope]
        # Also include values from start_m to min(111, baseline_scope) that we might have missed
        # Actually the list above covers 112, 111, 110, ..., start_m
    else:
        m_values = list(range(start_m, baseline_scope + 1))

    best_found = baseline_rows
    best_scope = baseline_scope

    for M in m_values:
        if time.time() > DEADLINE:
            break
        if M >= best_scope:
            # No point trying M >= best_scope unless it could improve
            # Actually if M >= best_scope, we already have a solution with scope <= best_scope
            # So skip
            continue

        solution = solve_optimized(n_rows, k, M)
        if solution is not None:
            best_found = solution
            best_scope = max(max(row) for row in solution)
            # If we found a solution at the lower bound, that's optimal
            if M == lower_bound:
                return solution

    return best_found

def greedy_construct(n_rows, k):
    """
    Fast greedy algorithm for baseline.
    """
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def solve_optimized(n_rows, k, M):
    """
    Backtracking solver with bitmask-based difference tracking.
    """
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # Fast popcount for Python 3.10+
    def popcount(x):
        return x.bit_count()

    # Precompute: for each possible row length position and current last value,
    # we could precompute valid next values, but that's complex. Instead, we use
    # strong pruning within the search.

    used_mask = 0
    rows = []
    solution_found = False

    def build_row(current_row, current_row_mask, rows_remaining, randomize_order):
        nonlocal used_mask, rows, solution_found

        if time.time() > DEADLINE:
            return False

        if len(current_row) == k + 1:
            # Row complete
            rows.append(list(current_row))
            used_mask |= current_row_mask
            return True

        # Pruning: total differences remaining must be >= needed
        used_count = popcount(used_mask)
        current_row_used = popcount(current_row_mask)
        # Diffs already committed: used_count (from completed rows)
        # Diffs in progress: current_row_used (from current row, not yet in used_mask)
        # Total diffs that will be used: used_count + current_row_used + (remaining diffs for current row) + (diffs for future rows)
        # We need: used_count + current_row_used + (diffs_per_row - current_row_used) + rows_remaining * diffs_per_row <= M
        # Simplifies to: used_count + diffs_per_row + rows_remaining * diffs_per_row <= M
        # i.e., used_count + (rows_remaining + 1) * diffs_per_row <= M

        remaining_rows_total = rows_remaining + 1  # including current
        if used_count + remaining_rows_total * diffs_per_row > M:
            return False

        last_val = current_row[-1]
        candidates = []
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                if d > M:
                    valid = False
                    break
                bit = 1 << d
                if (used_mask & bit) or (current_row_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            if valid:
                candidates.append((x, new_mask))

        if not candidates:
            return False

        # Heuristic ordering
        if randomize_order:
            random.shuffle(candidates)
        else:
            # Sort by a heuristic: prefer values that create smaller differences
            # (tighter packing), but also consider how many new differences they use
            # that are "rare" (high values are more constrained)
            # Simple: sort by x ascending
            pass

        for x, new_mask in candidates:
            current_row.append(x)
            if build_row(current_row, current_row_mask | new_mask, rows_remaining - 1, randomize_order):
                return True
            current_row.pop()

        return False

    def run_search(randomize):
        nonlocal used_mask, rows, solution_found
        used_mask = 0
        rows = []

        for i in range(n_rows):
            if time.time() > DEADLINE:
                return False
            if not build_row([0], 0, n_rows - i - 1, randomize):
                return False
        return True

    # Try deterministic first
    if run_search(False):
        return rows

    # Then randomized restarts
    for _ in range(8):
        if time.time() > DEADLINE:
            return None
        if run_search(True):
            return rows

    return None
