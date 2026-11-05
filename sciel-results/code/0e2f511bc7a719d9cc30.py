import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    total_diffs = n_rows * (k + 1) * k // 2

    def is_valid(rows):
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

    def get_scope(rows):
        return max(max(r) for r in rows)

    def greedy_construct(scope_limit, rng=None):
        """Greedy: pick smallest valid value at each step."""
        if rng is None:
            rng = random.Random(42)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                for x in range(last_val + 1, scope_limit + 1):
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

    def greedy_pressure(scope_limit, rng=None):
        """Pressure greedy: minimize the maximum new difference at each step."""
        if rng is None:
            rng = random.Random(42)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                best_x = None
                best_max_diff = float('inf')
                last_val = row[-1]
                for x in range(last_val + 1, scope_limit + 1):
                    max_d = 0
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        if d > max_d:
                            max_d = d
                    if valid and max_d < best_max_diff:
                        best_max_diff = max_d
                        best_x = x
                        # Early exit if we can't do better
                        if max_d == x - last_val:
                            break
                if best_x is None:
                    return None
                row.append(best_x)
                for y in row[:-1]:
                    used_diffs.add(best_x - y)
            rows.append(row)
        return rows

    def local_search(rows, time_budget, rng=None):
        """Local search to reduce scope: try moving individual elements."""
        if rng is None:
            rng = random.Random(42)
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])

        t_end = time.time() + time_budget
        no_improve_count = 0
        max_no_improve = 2000

        while time.time() < t_end and no_improve_count < max_no_improve:
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)  # Don't move position 0 (fixed at 0)

            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else current_scope + 5

            if min_val > max_val:
                no_improve_count += 1
                continue

            # Remove old diffs involving this position
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.append(d)
            for d in old_diffs:
                used_diffs.discard(d)

            # Try to find a better position
            best_new_val = None
            best_new_scope = current_scope
            # Try values that reduce scope (especially the max element)
            if pos == k:  # Last element, most impactful for scope
                search_range = range(min_val, min(max_val + 1, old_val))
            else:
                search_range = range(min_val, max_val + 1)

            for new_val in search_range:
                if new_val == old_val:
                    continue
                valid = True
                new_diffs = []
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = new_val - rows[row_idx][i]
                        else:
                            d = rows[row_idx][i] - new_val
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                if not valid:
                    continue
                # Check for duplicates within new_diffs
                if len(set(new_diffs)) != len(new_diffs):
                    continue

                # Compute new scope
                if pos == k:
                    new_row_max = new_val
                elif pos > 0:
                    new_row_max = max(rows[row_idx][pos + 1:]) if pos < k else new_val
                else:
                    new_row_max = max(rows[row_idx])

                other_max = 0
                for i in range(n_rows):
                    if i != row_idx:
                        other_max = max(other_max, max(rows[i]))
                    else:
                        if pos == k:
                            other_max = max(other_max, max(rows[i][:k]))
                        elif pos < k:
                            other_max = max(other_max, max(rows[i][:pos] + rows[i][pos+1:]))

                new_scope = max(new_row_max, other_max)

                if new_scope < best_new_scope:
                    best_new_scope = new_scope
                    best_new_val = new_val

            # Apply best found
            if best_new_val is not None and best_new_val != old_val:
                # Add new diffs
                for i in range(k + 1):
                    if i != pos:
                        if i < pos:
                            d = best_new_val - rows[row_idx][i]
                        else:
                            d = rows[row_idx][i] - best_new_val
                        used_diffs.add(d)

                rows[row_idx][pos] = best_new_val
                current_scope = best_new_scope

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Restore old diffs
                for d in old_diffs:
                    used_diffs.add(d)
                no_improve_count += 1

        return best_rows, best_scope

    def simulated_annealing(rows, time_budget, rng=None):
        """Simulated annealing to reduce scope."""
        if rng is None:
            rng = random.Random(42)
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])

        t_end = time.time() + time_budget
        initial_temp = max(1.0, current_scope / 3.0)
        current_temp = initial_temp
        cooling_rate = 0.9995
        no_improve = 0
        max_no_improve = 5000

        while time.time() < t_end and no_improve < max_no_improve:
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)

            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 10

            if min_val > max_val:
                current_temp *= cooling_rate
                continue

            # Generate candidate
            if pos == k:
                # Bias toward smaller values
                new_val = rng.randint(min_val, max(min_val + 20, max_val))
            else:
                new_val = rng.randint(min_val, max_val)

            if new_val == old_val:
                current_temp *= cooling_rate
                continue

            # Remove old diffs
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.append(d)
            for d in old_diffs:
                used_diffs.discard(d)

            # Check new diffs
            new_diffs = []
            valid = True
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = new_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - new_val
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
            if not valid or len(set(new_diffs)) != len(new_diffs):
                for d in old_diffs:
                    used_diffs.add(d)
                current_temp *= cooling_rate
                continue

            # Compute new scope
            if pos == k:
                new_row_max = new_val
            elif pos > 0:
                new_row_max = max(rows[row_idx][pos + 1:]) if pos < k else new_val
            else:
                new_row_max = max(rows[row_idx])

            other_max = 0
            for i in range(n_rows):
                if i != row_idx:
                    other_max = max(other_max, max(rows[i]))
                else:
                    if pos == k:
                        other_max = max(other_max, max(rows[i][:k]))
                    elif pos < k:
                        other_max = max(other_max, max(rows[i][:pos] + rows[i][pos+1:]))

            new_scope = max(new_row_max, other_max)
            delta = current_scope - new_scope  # positive = improvement

            # Acceptance
            if delta > 0 or rng.random() < math.exp(delta / current_temp):
                for d in new_diffs:
                    used_diffs.add(d)
                rows[row_idx][pos] = new_val
                current_scope = new_scope

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                for d in old_diffs:
                    used_diffs.add(d)
                no_improve += 1

            current_temp *= cooling_rate
            if current_temp < 0.01:
                current_temp = 0.01

        return best_rows, best_scope

    def backtracking_with_scope(scope_limit, time_budget, rng=None):
        """Backtracking to construct rows within a scope limit."""
        if rng is None:
            rng = random.Random(42)
        used_diffs = set()
        rows = []

        def try_build_row(current_row, time_limit):
            if time.time() > time_limit:
                return False

            if len(current_row) == k + 1:
                rows.append(current_row[:])
                return True

            last_val = current_row[-1]
            min_val = last_val + 1
            max_val = min(scope_limit, last_val + scope_limit)

            if min_val > max_val:
                return False

            # Try values, biased toward smaller ones
            candidates = list(range(min_val, max_val + 1))
            if len(candidates) > 30:
                # Mix: some small, some random
                small = candidates[:15]
                rest = candidates[15:]
                rand_sample = rng.sample(rest, min(20, len(rest)))
                candidates = small + rand_sample
                rng.shuffle(candidates[15:])
            else:
                rng.shuffle(candidates)

            for val in candidates:
                if time.time() > time_limit:
                    return False

                new_diffs = []
                valid = True
                for y in current_row:
                    d = val - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if not valid:
                    continue

                # Forward check
                remaining = k + 1 - len(current_row) - 1
                if remaining > 0 and val + remaining > scope_limit:
                    continue

                current_row.append(val)
                for d in new_diffs:
                    used_diffs.add(d)

                if try_build_row(current_row, time_limit):
                    return True

                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)

            return False

        time_limit = time.time() + time_budget
        for _ in range(n_rows):
            row = [0]
            if not try_build_row(row, time_limit):
                return None

        return rows

    # === Main strategy ===

    # Step 1: Get an initial solution using multiple greedy strategies
    initial_rows = None
    best_initial_scope = float('inf')

    strategies = [
        lambda s: greedy_construct(s, random.Random(42)),
        lambda s: greedy_pressure(s, random.Random(42)),
        lambda s: greedy_construct(s, random.Random(123)),
        lambda s: greedy_pressure(s, random.Random(123)),
        lambda s: greedy_construct(s, random.Random(777)),
        lambda s: greedy_pressure(s, random.Random(777)),
    ]

    for scope in [130, 135, 140, 150, 160, 180, 200, 300]:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        for strat in strategies:
            if time.time() > deadline - 5:
                break
            rows = strat(scope)
            if rows is not None and is_valid(rows):
                sc = get_scope(rows)
                if sc < best_initial_scope:
                    best_initial_scope = sc
                    initial_rows = rows

    if initial_rows is None:
        # Fallback: very loose greedy
        initial_rows = greedy_construct(2000, random.Random(42))
        if initial_rows is None:
            # Absolute fallback
            used_diffs = set()
            initial_rows = []
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
                initial_rows.append(row)
            return initial_rows

    current_scope = get_scope(initial_rows)

    # Step 2: Local search to reduce scope
    remaining = deadline - time.time()
    if remaining > 15:
        ls_rows, ls_scope = local_search(initial_rows, remaining * 0.4, random.Random(42))
        if ls_rows is not None and is_valid(ls_rows) and ls_scope < current_scope:
            initial_rows = ls_rows
            current_scope = ls_scope

    # Step 3: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 15:
        sa_rows, sa_scope = simulated_annealing(initial_rows, remaining * 0.4, random.Random(123))
        if sa_rows is not None and is_valid(sa_rows) and sa_scope < current_scope:
            initial_rows = sa_rows
            current_scope = sa_scope

    # Step 4: Try backtracking at a tighter scope if we have time
    remaining = deadline - time.time()
    if remaining > 20 and current_scope > 125:
        target = max(120, current_scope - 3)
        for seed in [42, 123, 777]:
            if time.time() > deadline - 5:
                break
            bt_rows = backtracking_with_scope(target, remaining * 0.3, random.Random(seed))
            if bt_rows is not None and is_valid(bt_rows):
                bt_scope = get_scope(bt_rows)
                if bt_scope < current_scope:
                    initial_rows = bt_rows
                    current_scope = bt_scope
                break

    # Step 5: Final local search polish
    remaining = deadline - time.time()
    if remaining > 10:
        final_rows, final_scope = local_search(initial_rows, remaining * 0.7, random.Random(999))
        if final_rows is not None and is_valid(final_rows) and final_scope < current_scope:
            initial_rows = final_rows
            current_scope = final_scope

    return initial_rows
