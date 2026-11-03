"""Difference triangle set (n_rows, k) constructor with randomized backtracking."""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # The problem: choose n_rows rows, each of k+1 increasing integers starting at 0,
    # such that all positive differences a_j - a_i (i<j) across all rows are distinct.
    # Minimize the largest entry.

    # Known best for (4,7) is 134. Simple counting bound is 112.
    # We'll use randomized backtracking search.

    best_result = None
    best_max_val = float('inf')

    # Number of restarts - we'll adapt based on time
    rng = random.Random(42)

    def is_valid_partial(rows_partial, used_diffs):
        """Check if partial assignment is valid (no duplicate diffs)."""
        # This is implicitly maintained, so just return True
        return True

    def solve_backtrack(n_rows, k, max_val, used_diffs, rows, row_idx, pos_in_row, current_row):
        """
        Backtracking solver.
        rows: list of completed rows
        row_idx: which row we're currently filling
        pos_in_row: position in current row (0 is already 0)
        current_row: the current row being built
        used_diffs: set of all differences used so far
        max_val: upper bound on largest entry
        Returns True if solution found, with rows filled.
        """
        if row_idx == n_rows:
            return True

        if pos_in_row == 0:
            current_row = [0]
            pos_in_row = 1

        if pos_in_row == k + 1:
            # Row complete
            rows.append(current_row)
            return solve_backtrack(n_rows, k, max_val, used_diffs, rows, row_idx + 1, 0, None)

        # Try to place next value
        # The next value must be > current_row[-1]
        start_val = current_row[-1] + 1
        if start_val > max_val:
            return False

        # For efficiency, we can try values in a randomized order
        # But we need to be careful about the search space
        # Let's try values from start_val to max_val
        # To speed up, we can use a heuristic: try smaller values first (greedy)
        # but with some randomization

        # Generate candidate values
        candidates = list(range(start_val, max_val + 1))

        # Randomize order slightly to escape local minima
        if len(candidates) > 1:
            # Keep first few in order, randomize the rest
            n_keep = min(3, len(candidates))
            rest = candidates[n_keep:]
            rng.shuffle(rest)
            candidates = candidates[:n_keep] + rest

        for val in candidates:
            # Check if this value creates any duplicate differences
            new_diffs = set()
            valid = True
            for a in current_row:
                d = val - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)

            if not valid:
                continue

            # Also check new_diffs don't have internal duplicates (shouldn't happen since current_row is strictly increasing)
            if len(new_diffs) != len(current_row):
                continue

            # Try this value
            current_row.append(val)
            used_diffs.update(new_diffs)

            if solve_backtrack(n_rows, k, max_val, used_diffs, rows, row_idx, pos_in_row + 1, current_row):
                return True

            # Backtrack
            used_diffs.difference_update(new_diffs)
            current_row.pop()

        return False

    # Strategy: try different max_val bounds from low to high
    # Counting bound: total differences = n_rows * k*(k+1)/2
    # These must all be distinct and <= max_val
    # So max_val >= n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs  # simple counting bound

    # For (4,7): 4*7*8/2 = 112
    # Known best is 134

    # We'll try bounds from lower_bound upward
    # But we need to be smart about it

    # First, let's try a greedy approach to get an initial upper bound
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Get initial upper bound from greedy
    greedy_rows = greedy_construct(n_rows, k)
    initial_max = max(max(row) for row in greedy_rows)
    best_result = greedy_rows
    best_max_val = initial_max

    # Now try to improve with backtracking
    # Try bounds from lower_bound to initial_max - 1
    for max_val in range(lower_bound, initial_max):
        if time.time() > deadline:
            break

        # Try multiple restarts with randomization
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            rng_seed = restart * 1000 + max_val
            local_rng = random.Random(rng_seed)
            # Monkey patch rng temporarily
            global_rng = rng
            rng = local_rng

            rows = []
            used_diffs = set()
            success = solve_backtrack(n_rows, k, max_val, used_diffs, rows, 0, 0, None)

            rng = global_rng

            if success:
                # Verify
                all_diffs = set()
                valid = True
                for row in rows:
                    if len(row) != k + 1:
                        valid = False
                        break
                    if row[0] != 0:
                        valid = False
                        break
                    for i in range(k + 1):
                        for j in range(i + 1, k + 1):
                            d = row[j] - row[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break

                if valid:
                    cur_max = max(max(row) for row in rows)
                    if cur_max < best_max_val:
                        best_max_val = cur_max
                        best_result = [list(row) for row in rows]
                    break  # Found solution for this max_val, try lower

    # Final check: ensure best_result is valid
    if best_result is not None:
        all_diffs = set()
        valid = True
        for row in best_result:
            if len(row) != k + 1:
                valid = False
                break
            if row[0] != 0:
                valid = False
                break
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        valid = False
                        break
                    all_diffs.add(d)
            if not valid:
                break

        if not valid:
            # Fallback to greedy
            best_result = greedy_construct(n_rows, k)

    return best_result
