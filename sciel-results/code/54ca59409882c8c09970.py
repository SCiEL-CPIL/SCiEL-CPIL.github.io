import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_diffs(row):
        """Get all differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
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

    def randomized_greedy(window_size=100):
        """
        Randomized greedy with window-based candidate selection.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last_val = row[-1]
                max_val = 5000
                found = False
                # Try a window of candidates with random shuffle
                candidates = list(range(last_val + 1, min(last_val + window_size, max_val + 1)))
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

    def count_conflicts(rows):
        """Count the number of duplicate differences across all rows."""
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        counts = Counter(diffs)
        conflicts = 0
        for count in counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

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

    def local_search(rows, time_limit):
        """
        Monotonic local search: only accept moves that strictly reduce scope.
        Uses a difference-based conflict detection for efficiency.
        """
        rows = [r[:] for r in rows]
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        current_scope = max(max(r) for r in rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        # Precompute for each position the set of differences it contributes
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

        t = start_time
        no_improve_count = 0

        while time.time() < t + time_limit:
            # Pick a random row and position
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)

            row = rows[row_idx]
            old_val = row[pos]
            min_val = row[pos - 1] + 1
            max_val = row[pos + 1] - 1 if pos < k else best_scope + 5

            if min_val > max_val:
                no_improve_count += 1
                continue

            # Try to find a new value that reduces scope
            # Prefer values closer to the lower bound
            best_new_val = None
            best_new_scope = current_scope

            # Sample several candidate values
            num_samples = min(30, max(1, max_val - min_val))
            if max_val - min_val <= 50:
                candidates = list(range(min_val, max_val + 1))
                random.shuffle(candidates)
                candidates = candidates[:num_samples]
            else:
                # Focus on lower values to reduce scope
                candidates = set()
                for _ in range(num_samples):
                    # Bias toward smaller values
                    bias = random.random() ** 0.6
                    candidates.add(min_val + int(bias * (max_val - min_val)))
                candidates = list(candidates)

            for new_val in candidates:
                if new_val == old_val:
                    continue

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
                    # Compute new scope
                    new_row_max = max(row[:pos] + [new_val] + row[pos + 1:])
                    other_max = max(max(r) for r in rows if r is not row) if n_rows > 1 else 0
                    new_scope = max(new_row_max, other_max)

                    if new_scope < best_new_scope:
                        best_new_scope = new_scope
                        best_new_val = new_val

                    # Restore and continue checking
                    used_diffs.update(old_diffs)
                else:
                    used_diffs.update(old_diffs)

            if best_new_val is not None and best_new_val != old_val:
                # Apply the best move
                old_diffs = position_diffs(row, pos)
                for d in old_diffs:
                    used_diffs.discard(d)

                row[pos] = best_new_val
                new_diffs = position_diffs(row, pos)
                used_diffs.update(new_diffs)

                current_scope = max(max(r) for r in rows)
                no_improve_count = 0

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                no_improve_count += 1

            # If stuck, try a more aggressive perturbation
            if no_improve_count > 150:
                # Randomly perturb a few positions
                for _ in range(8):
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
                                current_scope = max(max(r) for r in rows)
                            else:
                                used_diffs.update(old_diffs)

                no_improve_count = 0

        return best_rows, best_scope

    def simulated_annealing(initial_rows, time_limit):
        """
        Simulated annealing to reduce scope while maintaining validity.
        Combines exploration with exploitation.
        """
        rows = [r[:] for r in initial_rows]

        def compute_state(rows):
            used_diffs = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used_diffs.add(row[j] - row[i])
            scope = max(max(r) for r in rows)
            return used_diffs, scope

        used_diffs, current_scope = compute_state(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        def position_degree(row, pos):
            """Compute a heuristic degree for a position based on its differences."""
            degree = 0.0
            for i in range(k + 1):
                if i != pos:
                    d = abs(row[pos] - row[i])
                    if d > 0:
                        degree += 1.0 / d
            return degree

        t = start_time
        initial_temp = max(1.0, current_scope / 10.0)
        cooling_rate = 0.9998

        steps_since_improvement = 0
        max_steps_without_improvement = 400

        while time.time() < t + time_limit:
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]

            # Weighted position selection
            degrees = [0.0]
            for pos in range(1, k + 1):
                degrees.append(position_degree(row, pos))

            total_deg = sum(degrees[1:])
            if total_deg == 0:
                pos = random.randint(1, k)
            else:
                r = random.random() * total_deg
                cumulative = 0.0
                pos = k
                for p in range(1, k + 1):
                    cumulative += degrees[p]
                    if r <= cumulative:
                        pos = p
                        break

            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 10

            if min_val > max_val:
                continue

            # Prefer smaller new values
            if max_val - min_val > 50:
                bias = random.random() ** 0.7
                new_val = min_val + int(bias * (max_val - min_val))
            else:
                new_val = random.randint(min_val, max_val)

            if new_val == old_val:
                continue

            # Collect old diffs
            old_diffs = set()
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.add(d)

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
            new_scope = max(max(r) for r in rows if r is not rows[row_idx]) if n_rows > 1 else 0
            new_scope = max(new_scope, max(new_row))

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
                for _ in range(5):
                    r_idx = random.randint(0, n_rows - 1)
                    p_idx = random.randint(1, k)
                    min_v = rows[r_idx][p_idx - 1] + 1
                    max_v = rows[r_idx][p_idx + 1] - 1 if p_idx < k else best_scope + 10
                    if min_v <= max_v:
                        new_v = random.randint(min_v, max_v)
                        old_v = rows[r_idx][p_idx]
                        if new_v != old_v:
                            old_d = set()
                            for i in range(k + 1):
                                if i != p_idx:
                                    d = abs(old_v - rows[r_idx][i])
                                    old_d.add(d)
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
                                current_scope = max(max(r) for r in rows)
                            else:
                                used_diffs.update(old_d)

                steps_since_improvement = 0
                initial_temp = max(1.0, current_scope / 10.0)

        return best_rows, best_scope

    # Step 1: Generate diverse initial solutions
    best_rows = None
    best_scope = float('inf')

    # Try pure greedy first
    rows = greedy_construct()
    if rows is not None:
        scope = max(max(r) for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    # Try randomized greedy with multiple restarts and varying window sizes
    num_restarts = 12
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break
        window = 50 + restart * 20
        rows = randomized_greedy(window_size=window)
        if rows is not None:
            scope = max(max(r) for r in rows)
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
    if remaining_time > 20:
        ls_time = min(35, remaining_time - 15)
        best_rows, best_scope = local_search(best_rows, ls_time)

    # Step 3: Simulated annealing for further refinement
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        sa_time = remaining_time - 5
        best_rows, best_scope = simulated_annealing(best_rows, sa_time)

    # Final validation
    if not is_valid(best_rows):
        # If somehow invalid, fall back to greedy
        return greedy_construct()

    return best_rows
