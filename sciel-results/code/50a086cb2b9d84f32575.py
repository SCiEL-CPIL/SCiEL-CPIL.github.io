import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        """Get the maximum entry across all rows."""
        return max(max(r) for r in rows)

    def is_valid(rows):
        """Check if the rows form a valid DTS."""
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

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
                found = False
                last_val = row[-1]
                max_val = scope_limit if scope_limit is not None else 10000
                for x in range(last_val + 1, max_val + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def local_search(initial_rows, time_limit):
        """
        Structural change: Tabu search with short-term memory.
        Uses a boolean array for O(1) difference lookups.
        Move: change one element in one row.
        Tabu: remember (row_idx, pos, old_val) for last N moves.
        """
        rows = [r[:] for r in initial_rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        # Build used_diffs as a boolean array for O(1) lookup
        max_diff = current_scope
        used_diffs = [False] * (max_diff + 2)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_diff:
                        used_diffs[d] = True

        # Tabu list: (row_idx, pos, old_val) -> iteration when it becomes non-tabu
        tabu = {}
        tabu_tenure = 50
        iteration = 0

        t_end = time.time() + time_limit
        no_improve_count = 0
        max_no_improve = 200

        while time.time() < t_end:
            iteration += 1

            # Pick a random row and a random position to modify
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Positions 1..k (position 0 is always 0)

            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 10

            if min_val > max_val:
                continue

            # Check if this move is tabu
            tabu_key = (row_idx, pos, old_val)
            is_tabu = tabu.get(tabu_key, 0) > iteration

            # Try random new values, prefer values that reduce scope
            best_new_val = None
            best_new_scope = current_scope
            best_new_diffs = None
            best_is_tabu = False

            # Generate candidate values
            candidates = set()
            # Random candidates
            for _ in range(5):
                candidates.add(random.randint(min_val, max_val))
            # Candidates near old_val
            for delta in [-3, -2, -1, 1, 2, 3]:
                v = old_val + delta
                if min_val <= v <= max_val:
                    candidates.add(v)
            # If pos is last, try to reduce scope
            if pos == k:
                for delta in [-1, -2, -3, -5]:
                    v = old_val + delta
                    if min_val <= v <= max_val:
                        candidates.add(v)

            for new_val in candidates:
                if new_val == old_val:
                    continue

                # Collect old diffs involving position pos
                old_diffs = []
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = old_val - rows[row_idx][i]
                        else:
                            d = rows[row_idx][i] - old_val
                        old_diffs.append(d)

                # Temporarily remove old_diffs from used_diffs
                for d in old_diffs:
                    if d <= max_diff:
                        used_diffs[d] = False

                # Try adding new diffs
                new_diffs = []
                valid = True
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = new_val - rows[row_idx][i]
                        else:
                            d = rows[row_idx][i] - new_val
                        if d > max_diff:
                            # Need to expand the array
                            pass
                        if d <= max_diff and (used_diffs[d] or (d in new_diffs)):
                            valid = False
                            break
                        if d > max_diff:
                            # Check against existing new_diffs
                            if d in new_diffs:
                                valid = False
                                break
                        new_diffs.append(d)

                if not valid:
                    # Restore old_diffs
                    for d in old_diffs:
                        if d <= max_diff:
                            used_diffs[d] = True
                    continue

                # Compute new scope
                new_row_max = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
                other_max = 0
                for i in range(n_rows):
                    if i != row_idx:
                        if max(rows[i]) > other_max:
                            other_max = max(rows[i])
                new_scope = max(new_row_max, other_max)

                # Check if this move is tabu
                move_key = (row_idx, pos, new_val)
                move_is_tabu = tabu.get(move_key, 0) > iteration

                # Accept if better, or if not tabu
                if new_scope < best_new_scope or (new_scope == best_new_scope and not move_is_tabu and best_is_tabu):
                    best_new_scope = new_scope
                    best_new_val = new_val
                    best_new_diffs = new_diffs
                    best_is_tabu = move_is_tabu

            # Restore old_diffs if we didn't accept
            if best_new_val is None:
                for d in old_diffs:
                    if d <= max_diff:
                        used_diffs[d] = True
                continue

            # Check if the best move is tabu - only allow if it improves best
            if best_is_tabu and best_new_scope >= best_scope:
                # Reject tabu move that doesn't improve best
                for d in old_diffs:
                    if d <= max_diff:
                        used_diffs[d] = True
                continue

            # Accept the move
            rows[row_idx][pos] = best_new_val

            # Update used_diffs: remove old, add new
            for d in old_diffs:
                if d <= max_diff:
                    used_diffs[d] = False
            for d in best_new_diffs:
                if d <= max_diff:
                    used_diffs[d] = True

            current_scope = best_new_scope

            # Add to tabu list
            tabu[(row_idx, pos, old_val)] = iteration + tabu_tenure

            # Update best
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
            else:
                no_improve_count += 1

            # If no improvement for a while, diversify
            if no_improve_count > max_no_improve:
                # Reset tabu and try a different strategy
                tabu.clear()
                no_improve_count = 0
                max_no_improve = int(max_no_improve * 1.5)

            # Periodically clean up tabu
            if iteration % 1000 == 0:
                to_delete = [key for key, val in tabu.items() if val <= iteration]
                for key in to_delete:
                    del tabu[key]

        return best_rows, best_scope

    # Step 1: Try greedy construction with various scope limits
    rows = None
    for scope_limit in [100, 110, 112, 115, 120, 130, 140, 160, 200, 300, 500, 1000]:
        rows = greedy_construct(scope_limit)
        if rows is not None:
            break

    if rows is None:
        # Fallback: greedy without scope limit
        rows = greedy_construct()
        if rows is None:
            # Last resort: simple backtracking
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

    current_scope = get_scope(rows)

    # Step 2: Try to improve with local search (tabu)
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_rows, best_scope = local_search(rows, remaining_time - 5)
        if is_valid(best_rows):
            return best_rows
        # If local search produced invalid result, fall back to original
        return rows

    return rows
