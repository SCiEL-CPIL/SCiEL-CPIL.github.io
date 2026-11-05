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

    def simulated_annealing(initial_rows, time_limit):
        """
        Simulated annealing to reduce scope while maintaining validity.
        Uses a more aggressive cooling schedule and multiple restarts.
        """
        rows = [r[:] for r in initial_rows]
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        t_end = time.time() + time_limit
        initial_temp = max(1.0, current_scope / 3.0)
        cooling_rate = 0.9998

        # Precompute row maxima for faster scope calculation
        row_maxes = [max(r) for r in rows]
        global_max = max(row_maxes)

        while time.time() < t_end:
            # Pick a random row and a random position to modify
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Positions 1..k (position 0 is always 0)
            
            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 20
            
            if min_val > max_val:
                continue

            # Try a few random new values
            attempts = random.randint(1, 3)
            for _ in range(attempts):
                new_val = random.randint(min_val, max_val)
                if new_val == old_val:
                    continue

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
                new_row_max = max(rows[row_idx][pos - 1], new_val, rows[row_idx][pos + 1] if pos < k else 0)
                if new_row_max > row_maxes[row_idx]:
                    new_row_max = new_row_max
                else:
                    # Need to recompute max of the row if pos was the max
                    new_row_max = max(rows[row_idx])
                
                # Update row_maxes temporarily
                old_row_max = row_maxes[row_idx]
                row_maxes[row_idx] = max(rows[row_idx])
                
                # Compute new global scope
                if row_maxes[row_idx] > global_max:
                    new_scope = row_maxes[row_idx]
                else:
                    # Find max of other rows
                    new_scope = global_max
                    # Check if any other row has max
                    for i in range(n_rows):
                        if i != row_idx and row_maxes[i] > new_scope:
                            new_scope = row_maxes[i]
                
                # Accept or reject
                if new_scope < current_scope or random.random() < pow(2.718281828, (current_scope - new_scope) / initial_temp):
                    # Accept
                    rows[row_idx][pos] = new_val
                    used_diffs.update(new_diffs)
                    current_scope = new_scope
                    global_max = max(row_maxes)

                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]

                    break  # Move to next iteration of the while loop

                else:
                    # Reject: restore old_diffs and row_maxes
                    used_diffs.update(old_diffs)
                    row_maxes[row_idx] = old_row_max

            # Cool down
            initial_temp *= cooling_rate

        return best_rows, best_scope

    def local_search_polish(rows, time_limit):
        """
        Final high-precision polish pass: try to reduce scope by small integer steps.
        This acts as a local optimizer after SA.
        """
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        t_end = time.time() + time_limit

        # Precompute used diffs
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])

        # Identify which row has the maximum value
        max_row_idx = -1
        max_val = -1
        for i, r in enumerate(rows):
            if r[-1] > max_val:
                max_val = r[-1]
                max_row_idx = i

        # Try to reduce the last element of the max row, and also try reducing other elements
        while time.time() < t_end:
            if max_row_idx == -1:
                break
            
            # Try reducing the last element first
            row = rows[max_row_idx]
            last_pos = k
            old_val = row[last_pos]
            
            # Try decreasing the last element
            new_val = old_val - 1
            if new_val > row[last_pos - 1]:
                # Check validity
                old_diffs = set()
                for i in range(last_pos):
                    old_diffs.add(old_val - row[i])
                
                for d in old_diffs:
                    used_diffs.discard(d)
                
                valid = True
                new_diffs = set()
                for i in range(last_pos):
                    d = new_val - row[i]
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    row[last_pos] = new_val
                    used_diffs.update(new_diffs)
                    current_scope = get_scope(rows)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                else:
                    used_diffs.update(old_diffs)
            
            # Also try reducing other positions in the max row
            for pos in range(k - 1, 0, -1):
                old_val = row[pos]
                new_val = old_val - 1
                min_val = row[pos - 1] + 1
                max_val = row[pos + 1] - 1
                
                if new_val < min_val or new_val > max_val:
                    continue
                
                # Check validity
                old_diffs = set()
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = old_val - row[i]
                        else:
                            d = row[i] - old_val
                        old_diffs.add(d)
                
                for d in old_diffs:
                    used_diffs.discard(d)
                
                valid = True
                new_diffs = set()
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val
                        if d in used_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                
                if valid:
                    row[pos] = new_val
                    used_diffs.update(new_diffs)
                    current_scope = get_scope(rows)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                else:
                    used_diffs.update(old_diffs)
            
            # Update max_row_idx if needed
            max_row_idx = -1
            max_val = -1
            for i, r in enumerate(rows):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_row_idx = i

            # If we can't reduce the max row's last element, try other rows
            if max_row_idx != -1:
                row = rows[max_row_idx]
                if row[-1] == row[-1]:  # Check if we're stuck
                    # Try other rows
                    for other_idx in range(n_rows):
                        if other_idx == max_row_idx:
                            continue
                        other_row = rows[other_idx]
                        for pos in range(k, 0, -1):
                            old_val = other_row[pos]
                            new_val = old_val - 1
                            min_val = other_row[pos - 1] + 1
                            max_val = other_row[pos + 1] - 1 if pos < k else 100000
                            
                            if new_val < min_val or new_val > max_val:
                                continue
                            
                            # Check validity
                            old_diffs = set()
                            for i in range(k + 1):
                                if i != pos:
                                    if i < pos:
                                        d = old_val - other_row[i]
                                    else:
                                        d = other_row[i] - old_val
                                    old_diffs.add(d)
                            
                            for d in old_diffs:
                                used_diffs.discard(d)
                            
                            valid = True
                            new_diffs = set()
                            for i in range(k + 1):
                                if i != pos:
                                    if i < pos:
                                        d = new_val - other_row[i]
                                    else:
                                        d = other_row[i] - new_val
                                    if d in used_diffs or d in new_diffs:
                                        valid = False
                                        break
                                    new_diffs.add(d)
                            
                            if valid:
                                other_row[pos] = new_val
                                used_diffs.update(new_diffs)
                                current_scope = get_scope(rows)
                                if current_scope < best_scope:
                                    best_scope = current_scope
                                    best_rows = [r[:] for r in rows]
                                break
                            else:
                                used_diffs.update(old_diffs)

        return best_rows, best_scope

    # Step 1: Try greedy construction with various scope limits
    rows = None
    for scope_limit in [100, 120, 140, 160, 200, 300, 500, 1000]:
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

    # Step 2: Try to improve with simulated annealing
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Use more time for SA if we have plenty
        sa_time = min(remaining_time - 15, 90)
        if sa_time < 5:
            sa_time = remaining_time - 5
        best_rows, best_scope = simulated_annealing(rows, sa_time)
        if is_valid(best_rows):
            rows = best_rows
            current_scope = best_scope
        else:
            # If SA produced invalid result, fall back to original
            pass

    # Step 3: Final polish pass
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        polished_rows, polished_scope = local_search_polish(rows, remaining_time - 2)
        if is_valid(polished_rows):
            rows = polished_rows
            current_scope = polished_scope

    return rows
