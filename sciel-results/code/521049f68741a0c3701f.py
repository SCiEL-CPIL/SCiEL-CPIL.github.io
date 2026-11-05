import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def difference_pressure_greedy(seed=None):
        """
        Multi-start greedy with dynamic difference pressure ordering.
        Each row is built by selecting the next entry that minimizes the
        increase in the maximum used difference, naturally packing differences
        into the lower range.
        """
        if seed is not None:
            random.seed(seed)

        used_diffs = set()
        rows = []

        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last_val = row[-1]
                # Find candidates: values > last_val that don't conflict
                # Use difference pressure: prefer values that create smaller
                # new maximum differences
                best_candidate = None
                best_pressure = float('inf')
                # Search window: expand as needed
                search_limit = 2000
                for x in range(last_val + 1, search_limit + 1):
                    # Check all differences with existing row elements
                    valid = True
                    max_new_diff = 0
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        if d > max_new_diff:
                            max_new_diff = d
                    if valid:
                        # Pressure = how much this increases the overall max diff
                        current_max = max(used_diffs) if used_diffs else 0
                        pressure = max(max_new_diff, current_max) - current_max
                        # Add small random perturbation for diversity
                        pressure += random.random() * 0.1
                        if pressure < best_pressure:
                            best_pressure = pressure
                            best_candidate = x
                            # Early exit if we find a very good candidate
                            if pressure < 1.0:
                                break
                if best_candidate is None:
                    # Expand search
                    for x in range(search_limit + 1, 50000 + 1):
                        valid = True
                        for y in row:
                            d = x - y
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            best_candidate = x
                            break
                if best_candidate is None:
                    return None
                row.append(best_candidate)
                for y in row[:-1]:
                    used_diffs.add(best_candidate - y)
            rows.append(row)
        return rows

    def backtracking_solver(scope_limit):
        """
        Exact backtracking solver with constraint propagation.
        Tries to find a valid set with all entries <= scope_limit.
        """
        if k > 6:  # Too large for backtracking
            return None

        used_diffs = set()
        rows = []

        def backtrack(row_idx, pos, current_row):
            if pos == k + 1:
                rows.append(current_row[:])
                return True

            if row_idx > 0:
                min_val = current_row[-1] + 1 if current_row else 0
            else:
                min_val = 1 if pos > 0 else 0

            max_val = scope_limit
            if pos > 0 and current_row:
                min_val = current_row[-1] + 1

            for val in range(min_val, max_val + 1):
                if pos == 0 and val != 0:
                    continue
                # Check validity
                valid = True
                new_diffs = set()
                for i, y in enumerate(current_row):
                    d = val - y
                    if d <= 0:
                        continue
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)

                if valid:
                    # Forward check: can we still complete?
                    used_diffs.update(new_diffs)
                    current_row.append(val)
                    if backtrack(row_idx, pos + 1, current_row):
                        return True
                    current_row.pop()
                    used_diffs.difference_update(new_diffs)

            return False

        # Try each row
        for row_idx in range(n_rows):
            if not backtrack(row_idx, 0, []):
                return None
        return rows

    def local_search(rows, time_limit):
        """
        Local search to reduce scope by moving individual entries.
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

        def position_diffs(row, pos):
            diffs = set()
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        diffs.add(row[pos] - row[i])
                    else:
                        diffs.add(row[i] - row[pos])
            return diffs

        t = start_time
        no_improve = 0

        while time.time() < t + time_limit:
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            row = rows[row_idx]
            old_val = row[pos]
            min_val = row[pos - 1] + 1
            max_val = row[pos + 1] - 1 if pos < k else best_scope

            if min_val > max_val:
                no_improve += 1
                continue

            # Try to find a better (smaller) value
            best_new_val = None
            best_new_scope = current_scope

            # Sample candidates biased toward smaller values
            num_samples = min(30, max(1, max_val - min_val + 1))
            candidates = set()
            for _ in range(num_samples):
                bias = random.random() ** 0.5
                candidates.add(min_val + int(bias * (max_val - min_val)))
            candidates.add(min_val)

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
                    other_max = max(max(r) for r in rows if r is not row) if n_rows > 1 else 0
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
                used_diffs.update(position_diffs(row, pos))
                current_scope = max(max(r) for r in rows)
                no_improve = 0
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                no_improve += 1

            if no_improve > 300:
                # Perturbation
                for _ in range(5):
                    r_idx = random.randint(0, n_rows - 1)
                    p_idx = random.randint(1, k)
                    r = rows[r_idx]
                    min_v = r[p_idx - 1] + 1
                    max_v = r[p_idx + 1] - 1 if p_idx < k else best_scope + 5
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
                no_improve = 0

        return best_rows, best_scope

    def simulated_annealing(initial_rows, time_limit):
        """
        Simulated annealing for further scope reduction.
        """
        rows = [r[:] for r in initial_rows]
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        current_scope = max(max(r) for r in rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        t = start_time
        initial_temp = max(1.0, current_scope / 8.0)
        cooling_rate = 0.9997
        steps_since_improve = 0

        while time.time() < t + time_limit:
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]
            pos = random.randint(1, k)

            old_val = row[pos]
            min_val = row[pos - 1] + 1
            max_val = row[pos + 1] - 1 if pos < k else best_scope + 10

            if min_val > max_val:
                continue

            # Bias toward smaller values
            if max_val - min_val > 30:
                bias = random.random() ** 0.6
                new_val = min_val + int(bias * (max_val - min_val))
            else:
                new_val = random.randint(min_val, max_val)

            if new_val == old_val:
                continue

            # Compute old diffs
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

            new_diffs = set()
            valid = True
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

            if not valid:
                used_diffs.update(old_diffs)
                steps_since_improve += 1
                continue

            new_scope = max(new_val, max(row[:pos] + row[pos + 1:]))
            other_max = max(max(r) for r in rows if r is not row) if n_rows > 1 else 0
            new_scope = max(new_scope, other_max)

            if new_scope < current_scope:
                row[pos] = new_val
                used_diffs.update(new_diffs)
                current_scope = new_scope
                steps_since_improve = 0
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            elif random.random() < math.exp((current_scope - new_scope) / initial_temp):
                row[pos] = new_val
                used_diffs.update(new_diffs)
                current_scope = new_scope
                steps_since_improve += 1
            else:
                used_diffs.update(old_diffs)
                steps_since_improve += 1

            initial_temp *= cooling_rate

            if steps_since_improve > 400:
                # Perturbation
                for _ in range(3):
                    r_idx = random.randint(0, n_rows - 1)
                    p_idx = random.randint(1, k)
                    r = rows[r_idx]
                    min_v = r[p_idx - 1] + 1
                    max_v = r[p_idx + 1] - 1 if p_idx < k else best_scope + 8
                    if min_v <= max_v:
                        new_v = random.randint(min_v, max_v)
                        old_v = r[p_idx]
                        if new_v != old_v:
                            old_d = set()
                            for i in range(k + 1):
                                if i != p_idx:
                                    d = abs(old_v - r[i])
                                    old_d.add(d)
                            for d in old_d:
                                used_diffs.discard(d)
                            new_d = set()
                            valid_p = True
                            for i in range(k + 1):
                                if i != p_idx:
                                    if i < p_idx:
                                        d = new_v - r[i]
                                    else:
                                        d = r[i] - new_v
                                    if d in used_diffs or d in new_d:
                                        valid_p = False
                                        break
                                    new_d.add(d)
                            if valid_p:
                                r[p_idx] = new_v
                                used_diffs.update(new_d)
                                current_scope = max(max(r) for r in rows)
                            else:
                                used_diffs.update(old_d)
                steps_since_improve = 0
                initial_temp = max(1.0, current_scope / 8.0)

        return best_rows, best_scope

    # Step 1: Try backtracking for small instances
    if k <= 5:
        for scope in range(50, 200):
            rows = backtracking_solver(scope)
            if rows is not None:
                return rows

    # Step 2: Multi-start difference pressure greedy
    best_rows = None
    best_scope = float('inf')

    num_starts = 12
    for i in range(num_starts):
        if time.time() > deadline - 30:
            break
        rows = difference_pressure_greedy(seed=i * 12345 + 678)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        # Fallback: simple greedy
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

    # Step 3: Local search
    remaining = deadline - time.time()
    if remaining > 15:
        ls_time = min(25, remaining - 10)
        best_rows, best_scope = local_search(best_rows, ls_time)

    # Step 4: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 5:
        sa_time = remaining - 5
        best_rows, best_scope = simulated_annealing(best_rows, sa_time)

    return best_rows
