import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    best_max = float('inf')
    best_rows = None

    # --- Helper functions ---
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != row_len:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            diffs = get_diffs(row)
            if len(set(diffs)) != len(diffs):
                return False
            if any(d in used for d in diffs):
                return False
            used.update(diffs)
        return True

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # --- Greedy baseline ---
    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = max(max(r) for r in best_rows)

    # --- SA with efficient diff tracking ---
    def sa_optimize(initial_rows, time_limit, bound):
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len

        # Precompute diffs per row
        row_diffs = [get_diffs(r) for r in rows]

        # Global diff counts
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1

        def compute_cost():
            collisions = 0
            for d, c in diff_counts.items():
                if c > 1:
                    collisions += c - 1
            sum_entries = sum(sum(r) for r in rows)
            max_val = max(r[-1] for r in rows)
            return collisions * 10000 + sum_entries + max_val

        current_cost = compute_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = max(r[-1] for r in best_local_rows)

        T = 80.0
        cooling = 0.9998
        start_time = time.time()

        while time.time() - start_time < time_limit:
            r_idx = random.randint(0, n - 1)
            i_idx = random.randint(1, L - 1)

            row = rows[r_idx]
            old_val = row[i_idx]

            # Determine range
            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < L - 1 else bound + 10
            upper = min(upper, bound + 20)

            if lower > upper:
                continue

            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue

            # Remove old diffs for this row
            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                if d in diff_counts:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]

            # Apply change
            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs

            # Add new diffs
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1

            new_cost = compute_cost()

            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = max(r[-1] for r in best_local_rows)
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if random.random() < 0.5:
                        current_cost = new_cost
                    else:
                        # Revert
                        row[i_idx] = old_val
                        for d in new_diffs:
                            if d in diff_counts:
                                diff_counts[d] -= 1
                                if diff_counts[d] == 0:
                                    del diff_counts[d]
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row_diffs[r_idx] = old_diffs
                else:
                    if delta < 5000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0

                    if random.random() < p:
                        current_cost = new_cost
                    else:
                        # Revert
                        row[i_idx] = old_val
                        for d in new_diffs:
                            if d in diff_counts:
                                diff_counts[d] -= 1
                                if diff_counts[d] == 0:
                                    del diff_counts[d]
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row_diffs[r_idx] = old_diffs

            T *= cooling
            if T < 0.1:
                T = 0.1

        # High-precision polish pass
        T = 5.0
        cooling = 0.999
        polish_start = time.time()
        polish_limit = min(3.0, time_limit - (polish_start - start_time))

        # Reset to best found so far
        rows = [r[:] for r in best_local_rows]
        row_diffs = [get_diffs(r) for r in rows]
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        current_cost = compute_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = max(r[-1] for r in best_local_rows)

        while time.time() - polish_start < polish_limit:
            r_idx = random.randint(0, n - 1)
            i_idx = random.randint(1, L - 1)

            row = rows[r_idx]
            old_val = row[i_idx]

            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < L - 1 else best_local_max + 5
            upper = min(upper, best_local_max + 5)

            if lower > upper:
                continue

            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue

            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                if d in diff_counts:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]

            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs

            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1

            new_cost = compute_cost()

            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = max(r[-1] for r in best_local_rows)
            else:
                delta = new_cost - current_cost
                if delta < 200:
                    p = math.exp(-delta / T)
                else:
                    p = 0.0

                if random.random() < p:
                    current_cost = new_cost
                else:
                    row[i_idx] = old_val
                    for d in new_diffs:
                        if d in diff_counts:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                    for d in old_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    row_diffs[r_idx] = old_diffs

            T *= cooling
            if T < 0.1:
                T = 0.1

        return best_local_rows, best_local_max, best_local_cost

    # --- Main loop with multiple strategies ---
    while time.time() < deadline - 5.0:
        # Strategy 1: SA optimization
        sa_time = min(10.0, deadline - time.time() - 2.0)
        if sa_time <= 0:
            break

        # Initialize near best solution with slight randomization
        init_rows = [r[:] for r in best_rows]
        for _ in range(5):
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            row = init_rows[r_idx]
            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < row_len - 1 else best_max + 5
            if lower <= upper:
                init_rows[r_idx][i_idx] = random.randint(lower, upper)

        local_rows, local_max, local_cost = sa_optimize(init_rows, sa_time, best_max)

        if local_cost < 10000:  # Valid
            if local_max < best_max:
                best_rows = local_rows
                best_max = local_max

        # Strategy 2: Try greedy from scratch with different random seeds
        if time.time() < deadline - 10.0:
            for _ in range(3):
                greedy_rows = greedy_construct()
                greedy_max = max(max(r) for r in greedy_rows)
                if greedy_max < best_max:
                    best_rows = greedy_rows
                    best_max = greedy_max
                # Polish the greedy result
                polish_time = min(5.0, deadline - time.time() - 5.0)
                if polish_time > 0:
                    local_rows, local_max, local_cost = sa_optimize(greedy_rows, polish_time, best_max)
                    if local_cost < 10000:
                        if local_max < best_max:
                            best_rows = local_rows
                            best_max = local_max

    # Final validation
    if best_rows is None or not is_valid(best_rows):
        best_rows = greedy_construct()
        best_max = max(max(r) for r in best_rows)

    return best_rows
