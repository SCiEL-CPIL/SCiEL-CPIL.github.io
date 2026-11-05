import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    # Maximum possible difference we might need
    max_diff = 2000

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
                max_val = scope_limit if scope_limit is not None else max_diff
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

    def randomized_greedy():
        """
        Randomized greedy with restarts: build rows one by one, each row greedily
        but with some randomness in candidate selection to escape local optima.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last_val = row[-1]
                max_val = max_diff
                found = False
                # Try a window of candidates with random shuffle
                window_size = min(200, max_val - last_val)
                candidates = list(range(last_val + 1, last_val + window_size + 1))
                random.shuffle(candidates)
                for x in candidates:
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
                    # Fall back to linear scan
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

    def compute_scope(rows):
        return max(max(r) for r in rows)

    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def position_diffs(row, pos):
        """Get all differences involving position pos in this row."""
        diffs = set()
        for i in range(k + 1):
            if i != pos:
                if i < pos:
                    diffs.add(row[pos] - row[i])
                else:
                    diffs.add(row[i] - row[pos])
        return diffs

    def local_search_shrink(rows, time_limit):
        """
        Local search that specifically tries to reduce the scope (max entry).
        Focuses on the last element of each row first, then other positions.
        """
        rows = [r[:] for r in rows]
        used_diffs = compute_used_diffs(rows)
        current_scope = compute_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        t = start_time
        no_improve = 0

        while time.time() < t + time_limit:
            improved = False

            # Try to shrink the last element of each row
            for row_idx in range(n_rows):
                row = rows[row_idx]
                pos = k  # last position
                old_val = row[pos]
                min_val = row[pos - 1] + 1
                max_val = old_val - 1  # try to reduce

                if min_val > max_val:
                    continue

                # Try values from min_val upward (prefer smaller)
                for new_val in range(min_val, old_val):
                    # Check validity
                    old_diffs = position_diffs(row, pos)
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
                        current_scope = compute_scope(rows)
                        no_improve = 0
                        improved = True
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                        break
                    else:
                        used_diffs.update(old_diffs)

                if improved:
                    break

            if not improved:
                # Try other positions
                row_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                row = rows[row_idx]
                old_val = row[pos]
                min_val = row[pos - 1] + 1
                max_val = row[pos + 1] - 1 if pos < k else best_scope + 5

                if min_val > max_val:
                    no_improve += 1
                    continue

                # Try to find a new value that reduces scope
                best_new_val = None
                best_new_scope = current_scope

                # Sample several candidate values, biased toward smaller
                num_samples = min(30, max(1, max_val - min_val))
                if max_val - min_val <= 60:
                    candidates = list(range(min_val, max_val + 1))
                    random.shuffle(candidates)
                    candidates = candidates[:num_samples]
                else:
                    candidates = set()
                    for _ in range(num_samples):
                        bias = random.random() ** 0.5  # bias toward smaller
                        candidates.add(min_val + int(bias * (max_val - min_val)))
                    candidates = list(candidates)

                for new_val in candidates:
                    if new_val == old_val:
                        continue

                    old_diffs = position_diffs(row, pos)
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
                        new_row_max = max(row[:pos] + [new_val] + row[pos + 1:])
                        other_max = 0
                        for r in rows:
                            if r is not row:
                                other_max = max(other_max, max(r))
                        new_scope = max(new_row_max, other_max)

                        if new_scope < best_new_scope:
                            best_new_scope = new_scope
                            best_new_val = new_val

                        used_diffs.update(old_diffs)
                    else:
                        used_diffs.update(old_diffs)

                if best_new_val is not None and best_new_val != old_val:
                    old_diffs = position_diffs(row, pos)
                    for d in old_diffs:
                        used_diffs.discard(d)

                    row[pos] = best_new_val
                    new_diffs = position_diffs(row, pos)
                    used_diffs.update(new_diffs)

                    current_scope = compute_scope(rows)
                    no_improve = 0
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                else:
                    no_improve += 1

            # If stuck, try perturbation
            if no_improve > 100:
                for _ in range(3):
                    r_idx = random.randint(0, n_rows - 1)
                    p_idx = random.randint(1, k)
                    r = rows[r_idx]
                    min_v = r[p_idx - 1] + 1
                    max_v = r[p_idx + 1] - 1 if p_idx < k else best_scope + 10
                    if min_v <= max_v:
                        new_v = random.randint(min_v, max_v)
                        old_v = r[p_idx]
                        if new_v != old_v:
                            old_diffs = position_diffs(r, p_idx)
                            for d in old_diffs:
                                used_diffs.discard(d)

                            valid = True
                            new_diffs = set()
                            for i in range(k + 1):
                                if i != p_idx:
                                    if i < p_idx:
                                        d = new_v - r[i]
                                    else:
                                        d = r[i] - new_v
                                    if d in used_diffs or d in new_diffs:
                                        valid = False
                                        break
                                    new_diffs.add(d)

                            if valid:
                                r[p_idx] = new_v
                                used_diffs.update(new_diffs)
                                current_scope = compute_scope(rows)
                            else:
                                used_diffs.update(old_diffs)

                no_improve = 0

        return best_rows, best_scope

    def simulated_annealing(initial_rows, time_limit):
        """
        Simulated annealing to reduce scope while maintaining validity.
        """
        rows = [r[:] for r in initial_rows]
        used_diffs = compute_used_diffs(rows)
        current_scope = compute_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        t = start_time
        initial_temp = max(2.0, current_scope / 5.0)
        cooling_rate = 0.9998

        steps_since_improvement = 0
        max_steps_without_improvement = 300

        while time.time() < t + time_limit:
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]

            # Weighted position selection: bias toward last position
            if random.random() < 0.4:
                pos = k
            else:
                pos = random.randint(1, k)

            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 10

            if min_val > max_val:
                continue

            # Prefer smaller new values
            if max_val - min_val > 50:
                bias = random.random() ** 0.6
                new_val = min_val + int(bias * (max_val - min_val))
            else:
                new_val = random.randint(min_val, max_val)

            if new_val == old_val:
                continue

            # Collect old diffs
            old_diffs = position_diffs(row, pos)
            for d in old_diffs:
                used_diffs.discard(d)

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
                used_diffs.update(old_diffs)
                steps_since_improvement += 1
                continue

            new_row = rows[row_idx][:]
            new_row[pos] = new_val
            other_max = 0
            for r in rows:
                if r is not rows[row_idx]:
                    other_max = max(other_max, max(r))
            new_scope = max(other_max, max(new_row))

            if new_scope < current_scope:
                rows[row_idx][pos] = new_val
                used_diffs.update(new_diffs)
                current_scope = new_scope
                steps_since_improvement = 0

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]

            elif random.random() < math.exp((current_scope - new_scope) / initial_temp):
                rows[row_idx][pos] = new_val
                used_diffs.update(new_diffs)
                current_scope = new_scope
                steps_since_improvement += 1
            else:
                used_diffs.update(old_diffs)
                steps_since_improvement += 1

            initial_temp *= cooling_rate

            if steps_since_improvement > max_steps_without_improvement:
                # Perturbation
                for _ in range(5):
                    r_idx = random.randint(0, n_rows - 1)
                    p_idx = random.randint(1, k)
                    min_v = rows[r_idx][p_idx - 1] + 1
                    max_v = rows[r_idx][p_idx + 1] - 1 if p_idx < k else best_scope + 10
                    if min_v <= max_v:
                        new_v = random.randint(min_v, max_v)
                        old_v = rows[r_idx][p_idx]
                        if new_v != old_v:
                            old_d = position_diffs(rows[r_idx], p_idx)
                            for d in old_d:
                                used_diffs.discard(d)

                            new_d = set()
                            valid_perturb = True
                            for i in range(k + 1):
                                if i != p_idx:
                                    if i < p_idx:
                                        d = new_v - rows[r_idx][i]
                                    else:
                                        d = rows[r_idx][i] - new_v
                                    if d in used_diffs or d in new_d:
                                        valid_perturb = False
                                        break
                                    new_d.add(d)

                            if valid_perturb:
                                rows[r_idx][p_idx] = new_v
                                used_diffs.update(new_d)
                                current_scope = compute_scope(rows)
                            else:
                                used_diffs.update(old_d)

                steps_since_improvement = 0
                initial_temp = max(2.0, current_scope / 5.0)

        return best_rows, best_scope

    def backtracking_search(n_rows, k, scope, used_diffs, rows, current_row, current_pos):
        """
        Backtracking search to find a valid configuration within a given scope.
        Returns True if found, with rows populated.
        """
        if current_row is not None and current_pos < k + 1:
            last_val = current_row[-1]
            min_val = last_val + 1
            max_val = scope
            if min_val > max_val:
                return False

            # Try values in increasing order
            for x in range(min_val, max_val + 1):
                valid = True
                for y in current_row:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    new_row = current_row + [x]
                    new_used = used_diffs | set(x - y for y in current_row)
                    if len(new_used) > len(used_diffs) + len(current_row):
                        # Check for duplicates within new differences
                        pass

                    if backtracking_search(n_rows, k, scope, new_used, rows, new_row, current_pos + 1):
                        return True
            return False
        elif current_row is not None and current_pos == k + 1:
            # Completed a row
            new_rows = rows + [current_row]
            if len(new_rows) == n_rows:
                return True
            return backtracking_search(n_rows, k, scope, used_diffs, new_rows, None, 0)
        else:
            # Starting a new row
            if backtracking_search(n_rows, k, scope, used_diffs, rows, [0], 1):
                return True
            return False

    # Step 1: Generate diverse initial solutions
    best_rows = None
    best_scope = float('inf')

    # Try pure greedy first
    rows = greedy_construct()
    if rows is not None:
        scope = compute_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    # Try randomized greedy with multiple restarts
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        rows = randomized_greedy()
        if rows is not None:
            scope = compute_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        # Fallback: simple construction
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

    # Step 2: Local search for quick improvement
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        ls_time = min(25, remaining_time - 10)
        best_rows, best_scope = local_search_shrink(best_rows, ls_time)

    # Step 3: Simulated annealing for further refinement
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        sa_time = remaining_time - 5
        best_rows, best_scope = simulated_annealing(best_rows, sa_time)

    return best_rows
