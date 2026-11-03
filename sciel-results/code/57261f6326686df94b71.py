import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    best_max = float('inf')
    best_rows = None

    # --- Helper Functions ---

    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def compute_max(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        return len(all_diffs) == len(set(all_diffs))

    # --- Greedy Baseline ---
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

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = compute_max(best_rows)

    # --- Multi-Start Simulated Annealing ---

    def sa_optimize(initial_rows, time_limit, seed_offset=0):
        random.seed(seed_offset)
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len

        # Precompute diffs per row and global counts
        row_diffs = [get_diffs(r) for r in rows]
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1

        # Calculate initial collision count
        collision_count = 0
        for d, c in diff_counts.items():
            if c > 1:
                collision_count += c - 1

        current_max = compute_max(rows)
        W_COLL = 10000

        def get_cost():
            return collision_count * W_COLL + current_max

        current_cost = get_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max

        T = 200.0
        cooling = 0.99995
        min_T = 0.5
        start_time = time.time()

        while time.time() - start_time < time_limit:
            # Select random row and index
            r_idx = random.randint(0, n - 1)
            i_idx = random.randint(1, L - 1)

            row = rows[r_idx]
            old_val = row[i_idx]

            # Determine valid range for new_val
            lower = row[i_idx - 1] + 1
            if i_idx < L - 1:
                upper = row[i_idx + 1] - 1
            else:
                upper = current_max + 30

            if lower > upper:
                continue

            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue

            # --- Update State ---

            # 1. Remove old diffs
            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                c = diff_counts[d]
                if c == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] = c - 1
                    if c >= 2:
                        collision_count -= 1

            # 2. Update row
            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs

            # 3. Add new diffs
            for d in new_diffs:
                c = diff_counts.get(d, 0)
                if c == 0:
                    diff_counts[d] = 1
                else:
                    diff_counts[d] = c + 1
                    if c >= 1:
                        collision_count += 1

            # 4. Update Max
            new_global_max = compute_max(rows)

            new_cost = collision_count * W_COLL + new_global_max

            # --- Acceptance ---
            delta = new_cost - current_cost
            accept = False

            if delta < 0:
                accept = True
            else:
                if T > min_T:
                    p = math.exp(-delta / T)
                    if random.random() < p:
                        accept = True
                elif delta == 0:
                    if random.random() < 0.3:
                        accept = True

            if accept:
                current_cost = new_cost
                current_max = new_global_max
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = new_global_max
            else:
                # Revert
                row[i_idx] = old_val
                row_diffs[r_idx] = old_diffs

                # Re-add old diffs
                for d in old_diffs:
                    c = diff_counts.get(d, 0)
                    if c == 0:
                        diff_counts[d] = 1
                    else:
                        diff_counts[d] = c + 1
                        if c >= 1:
                            collision_count += 1

                # Remove new diffs
                for d in new_diffs:
                    c = diff_counts[d]
                    if c == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = c - 1
                        if c >= 2:
                            collision_count -= 1

                current_max = compute_max(rows)

            T *= cooling
            if T < min_T:
                T = min_T

        return best_local_rows, best_local_max

    # --- Polish Phase: try to reduce the maximum entry ---
    def polish(rows, time_limit):
        rows = [r[:] for r in rows]
        current_max = compute_max(rows)
        start_time = time.time()

        diff_counts = {}
        for r in rows:
            for d in get_diffs(r):
                diff_counts[d] = diff_counts.get(d, 0) + 1

        # Check if valid
        for d, c in diff_counts.items():
            if c > 1:
                return rows, current_max

        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try to reduce the last element of each row
            for r_idx in range(n_rows):
                row = rows[r_idx]
                old_last = row[-1]
                # Try values from old_last-1 down to row[-2]+1
                for new_val in range(old_last - 1, row[-2], -1):
                    old_diffs_to_remove = [old_last - x for x in row[:-1]]
                    new_diffs_to_add = [new_val - x for x in row[:-1]]

                    # Check if new diffs are all unused
                    valid = True
                    for d in new_diffs_to_add:
                        if d in diff_counts:
                            valid = False
                            break

                    if valid:
                        # Remove old diffs
                        for d in old_diffs_to_remove:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        # Add new diffs
                        for d in new_diffs_to_add:
                            diff_counts[d] = 1
                        row[-1] = new_val
                        improved = True
                        current_max = compute_max(rows)
                        break

        return rows, current_max

    # --- Main Loop: Multi-Start SA ---
    elapsed = time.time() - time.time()
    remaining = deadline - time.time()

    # Do multiple SA runs with different seeds
    n_starts = 5
    time_per_start = remaining / n_starts

    for s in range(n_starts):
        if time.time() >= deadline - 5.0:
            break

        # Use a perturbed version of the best known solution
        if s == 0:
            start_rows = best_rows
        else:
            # Perturb the best solution
            start_rows = [r[:] for r in best_rows]
            for _ in range(3):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                old_val = start_rows[r_idx][i_idx]
                lower = start_rows[r_idx][i_idx - 1] + 1
                if i_idx < row_len - 1:
                    upper = start_rows[r_idx][i_idx + 1] - 1
                else:
                    upper = compute_max(start_rows) + 10
                if lower <= upper:
                    start_rows[r_idx][i_idx] = random.randint(lower, upper)

        t_remaining = deadline - time.time() - 5.0
        if t_remaining <= 0:
            break

        local_rows, local_max = sa_optimize(start_rows, t_remaining, seed_offset=42 + s * 1000)

        # Check validity
        if is_valid(local_rows):
            if local_max < best_max:
                best_rows = local_rows
                best_max = local_max

    # Final polish
    if time.time() < deadline - 1.0:
        polish_time = deadline - time.time() - 1.0
        polished_rows, polished_max = polish(best_rows, polish_time)
        if polished_max < best_max:
            best_rows = polished_rows
            best_max = polished_max

    return best_rows
