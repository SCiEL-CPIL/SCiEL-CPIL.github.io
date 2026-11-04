import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    # We'll use a set of used differences for clarity, but for speed we can use a set of ints
    # or a boolean array. Since scope is at most a few hundred to a few thousand,
    # a set of ints is fine.

    def is_valid_row(row, used_diffs):
        """Check if row has all distinct differences and none conflict with used_diffs."""
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        # Check all differences
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs or d in row_diffs:
                    return False
                row_diffs.add(d)
        return True

    def get_diffs(row):
        """Get all differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_construct(scope_limit=None):
        """
        Greedy construction: build rows one by one, each row greedily.
        Returns list of rows or None if stuck.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Try to add next element
                found = False
                # Start from last_val + 1, go up
                last_val = row[-1]
                # If scope_limit, don't exceed it
                max_val = scope_limit if scope_limit is not None else 10000
                for x in range(last_val + 1, max_val + 1):
                    # Check if x can be added
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        # Also check within-row uniqueness (implicitly handled since we check against used_diffs
                        # which doesn't include current row's diffs yet, but we need to ensure new diffs are unique
                        # within the row too)
                        # Actually, we need to check that the new differences (x - y for y in row)
                        # are all distinct from each other AND not in used_diffs.
                        # Since y are all distinct and x is fixed, x-y are all distinct.
                        # So we just need to check against used_diffs.
                        row.append(x)
                        # Update used_diffs with new differences
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def simulated_annealing(initial_rows, time_limit):
        """
        Simulated annealing to reduce scope while maintaining validity.
        """
        rows = [r[:] for r in initial_rows]
        # Compute current used_diffs and scope
        def compute_state(rows):
            used_diffs = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        used_diffs.add(row[j] - row[i])
            scope = max(max(r) for r in rows)
            return used_diffs, scope

        used_diffs, current_scope = compute_state(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        # Temperature schedule
        t = start_time
        initial_temp = max(1.0, current_scope / 10.0)
        cooling_rate = 0.999  # Adjust based on steps

        while time.time() < t + time_limit:
            # Pick a random row and a random position to modify
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Positions 1..k (position 0 is always 0)
            
            old_val = rows[row_idx][pos]
            # Try to change this value
            # New value should be > previous element and < next element (if exists)
            min_val = rows[row_idx][pos-1] + 1
            max_val = rows[row_idx][pos+1] - 1 if pos < k else best_scope + 10
            
            if min_val > max_val:
                continue

            # Random new value
            new_val = random.randint(min_val, max_val)
            if new_val == old_val:
                continue

            # Check if changing this value maintains validity
            # We need to:
            # 1. Remove old differences involving rows[row_idx][pos]
            # 2. Add new differences
            # 3. Check for conflicts

            # Collect old diffs involving position pos
            old_diffs = set()
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.add(d)

            # Temporarily remove old_diffs from used_diffs
            # But we need to be careful: some of these diffs might be shared with other rows?
            # No, differences are unique across all rows, so each diff belongs to exactly one row.
            # So we can safely remove them.
            for d in old_diffs:
                used_diffs.discard(d)

            # Try adding new diffs
            new_diffs = set()
            valid = True
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = new_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - new_val
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)

            if not valid:
                # Restore old_diffs
                used_diffs.update(old_diffs)
                continue

            # Compute new scope
            new_row = rows[row_idx][:]
            new_row[pos] = new_val
            new_scope = max(max(r) for r in rows if r is not rows[row_idx]) if n_rows > 1 else 0
            new_scope = max(new_scope, max(new_row))

            # Accept or reject
            if new_scope < current_scope or random.random() < pow(2.718281828, (current_scope - new_scope) / initial_temp):
                # Accept
                rows[row_idx][pos] = new_val
                used_diffs.update(new_diffs)
                current_scope = new_scope

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]

            else:
                # Reject: restore old_diffs
                used_diffs.update(old_diffs)

            # Cool down
            initial_temp *= cooling_rate

        return best_rows, best_scope

    # Step 1: Greedy construction
    rows = greedy_construct()
    if rows is None:
        # Fallback: try with a larger scope
        rows = greedy_construct(scope_limit=5000)
        if rows is None:
            # Last resort: simple backtracking for small cases
            # For now, return a trivial solution
            used_diffs = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for y in row:
                        if x - y in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                rows.append(row)
            return rows

    current_scope = max(max(r) for r in rows)

    # Step 2: Simulated annealing to improve
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_rows, best_scope = simulated_annealing(rows, remaining_time - 5)
        return best_rows

    return rows
