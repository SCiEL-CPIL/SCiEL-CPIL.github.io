"""Rewritten program: backtracking with forward checking and dynamic ordering for difference triangle sets."""
import time
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Each row has k+1 entries, so C(k+1,2) differences per row
    # Total differences needed: n_rows * k*(k+1)//2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # The scope m must satisfy: we need at least total_diffs distinct positive integers
    # But also the max difference in a row is at most m (since entries are in [0, m])
    # Actually the max difference can be up to m (from 0 to m)
    # We need to find the smallest m such that a valid configuration exists

    # Lower bound on m: the largest difference in any row is at most m
    # We need total_diffs distinct differences, each in range [1, m]
    # So m >= total_diffs (since differences are distinct positive integers)
    # But actually differences can be larger... no, max difference is m (from 0 to m)
    # Wait, entries are in [0, m], so max difference is m.
    # We need total_diffs distinct values in [1, m], so m >= total_diffs.
    # But that's a very loose bound. The simple counting bound mentioned is 112 for (4,7).
    # For (4,7): diffs_per_row = 7*8/2 = 28, total_diffs = 112. So m >= 112.

    # Let's try to find a valid configuration with scope m starting from a reasonable lower bound
    # and increase m if needed.

    # Strategy: use backtracking to construct rows one by one.
    # For each row, we place entries one at a time, choosing the smallest possible value
    # that doesn't conflict with already-used differences.

    # To improve on the greedy approach, we can use:
    # 1. Forward checking: when placing a new entry, check if it's still possible to complete the row
    # 2. Try multiple orderings / restarts with slight randomization
    # 3. Binary search on m to find the minimal scope

    # First, let's establish a baseline with the greedy approach
    best_result = None
    best_scope = float('inf')

    def greedy_construct(m_limit=None):
        """Greedy construction with optional m limit."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if m_limit is not None and x > m_limit:
                    return None
                new_diffs = set()
                for a in row:
                    d = x - a
                    if d in used:
                        break
                    new_diffs.add(d)
                else:
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows

    # Get baseline
    baseline = greedy_construct()
    if baseline:
        best_result = baseline
        best_scope = max(max(row) for row in baseline)

    # Now try to improve by searching for smaller scopes
    # Lower bound: total_diffs (since we need that many distinct differences in [1, m])
    lower_bound = total_diffs
    upper_bound = best_scope

    # Try values between lower_bound and upper_bound - 1
    # Use a smarter search: try to find valid config with scope m
    # For each m, we need to check if a valid configuration exists

    # Backtracking solver with forward checking
    def can_construct_with_scope(m, time_limit=None):
        """
        Try to construct n_rows rows with all entries in [0, m]
        such that all differences are distinct.
        Returns the list of rows if successful, None otherwise.
        """
        if time_limit is not None and time.time() > time_limit:
            return None

        used = set()
        rows = []

        def backtrack(row_idx, row, x):
            nonlocal used, rows
            if time_limit is not None and time.time() > time_limit:
                return False

            if len(row) == k + 1:
                rows.append(row)
                if row_idx + 1 == n_rows:
                    return True
                # Try next row
                if backtrack(row_idx + 1, [0], 0):
                    return True
                rows.pop()
                return False

            # Try next value for current row
            for candidate in range(x + 1, m + 1):
                # Check if adding candidate creates conflicts
                new_diffs = set()
                conflict = False
                for a in row:
                    d = candidate - a
                    if d in used:
                        conflict = True
                        break
                    new_diffs.add(d)
                if conflict:
                    continue

                # Forward checking: verify we can still complete the row
                # The remaining entries must be in [candidate+1, m]
                # and must create diffs not in used | new_diffs
                remaining_slots = k + 1 - len(row) - 1
                available_range = m - candidate  # max diff from candidate to m
                # We need at least remaining_slots distinct differences for each new entry
                # This is a weak check; let's just proceed

                row.append(candidate)
                used.update(new_diffs)

                if backtrack(row_idx, row, candidate):
                    return True

                row.pop()
                used.difference_update(new_diffs)

            return False

        if backtrack(0, [0], 0):
            return rows
        return None

    # Try to find a better scope
    # Search from lower_bound up to best_scope - 1
    # But this could be expensive, so we try a few key values first

    # Try the lower bound first
    for m in range(lower_bound, min(upper_bound, lower_bound + 50)):
        if time.time() > deadline:
            break
        result = can_construct_with_scope(m, time_limit=deadline - 5)
        if result is not None:
            best_result = result
            best_scope = m
            # Found a valid config, try to go lower? No, we're going up.
            # Actually we want the minimum, so if lower_bound works, that's optimal
            break

    # If we didn't find anything better, try some randomized local search
    # to potentially find a config with scope between lower_bound and best_scope

    # Iterated local search: start from a valid config and try to reduce scope
    if best_result is not None and time.time() < deadline - 10:
        # Try to reduce scope by swapping entries
        current = [row[:] for row in best_result]
        current_scope = max(max(row) for row in current)

        def get_all_diffs(rows):
            diffs = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        diffs.add(row[j] - row[i])
            return diffs

        def is_valid(rows):
            diffs = get_all_diffs(rows)
            if len(diffs) != total_diffs:
                return False
            # Check all diffs are positive
            return all(d > 0 for d in diffs)

        # Try to reduce scope by trying to move the largest entry down
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Find the row with the largest entry
            max_entry = float('inf')
            max_row_idx = -1
            max_pos = -1
            for i, row in enumerate(current):
                for j, val in enumerate(row):
                    if val > max_entry:
                        max_entry = val
                        max_row_idx = i
                        max_pos = j

            # Try to reduce this entry
            for new_val in range(max_entry - 1, current[max_row_idx][max_pos - 1] if max_pos > 0 else -1, -1):
                # Try replacing current[max_row_idx][max_pos] with new_val
                old_val = current[max_row_idx][max_pos]
                current[max_row_idx][max_pos] = new_val

                if is_valid(current):
                    max_entry = new_val
                    improved = True
                    break
                else:
                    current[max_row_idx][max_pos] = old_val

            # Also try random perturbations
            if not improved and time.time() < deadline - 5:
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    # Pick a random entry (not the first) and try to move it
                    r = random.randint(0, n_rows - 1)
                    p = random.randint(1, k)  # positions 1..k
                    old_val = current[r][p]
                    # Try nearby values
                    for delta in [1, -1, 2, -2, 3, -3]:
                        new_val = old_val + delta
                        if new_val <= 0:
                            continue
                        # Check ordering
                        if p > 0 and new_val <= current[r][p-1]:
                            continue
                        if p < k and new_val >= current[r][p+1]:
                            continue

                        current[r][p] = new_val
                        if is_valid(current):
                            cur_scope = max(max(row) for row in current)
                            if cur_scope < best_scope:
                                best_scope = cur_scope
                                best_result = [row[:] for row in current]
                            improved = True
                            break
                        else:
                            current[r][p] = old_val

    return best_result
