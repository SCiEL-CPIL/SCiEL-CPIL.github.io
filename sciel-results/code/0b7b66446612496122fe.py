import time
import random
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: greedy initialization, SA local search, and iterative
    row-level re-optimization with a priority-queue-based conflict-aware selection.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # ---- Greedy fallback (always valid) ----
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_used(rows):
        all_used = set()
        for row in rows:
            all_used |= _row_diffs(row)
        return all_used

    def _global_max(rows):
        return max(max(row) for row in rows)

    def local_search_init(n_rows, k, seed=None):
        """Greedy initialization."""
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def _reoptimize_row(row, other_diffs, k, max_scope=None):
        """
        Globally re-optimize a single row against fixed differences from other rows.
        Uses iterative local search on the row elements to minimize the max entry.
        """
        n = k + 1
        best_row = list(row)
        best_max = max(best_row)

        if max_scope is not None and best_max > max_scope:
            return best_row

        other_set = set(other_diffs)

        for iteration in range(120):
            improved = False
            # Process positions in order of largest value first (greedy compression)
            order = sorted(range(1, n), key=lambda i: -best_row[i])
            for pi in order:
                old_val = best_row[pi]
                min_allowed = best_row[pi - 1] + 1
                max_allowed = (best_row[pi + 1] - 1) if pi < n - 1 else (max_scope - 1 if max_scope else old_val)

                # Try decreasing values first
                for new_val in range(old_val - 1, min_allowed - 1, -1):
                    if new_val < min_allowed:
                        break
                    if pi < n - 1 and new_val >= best_row[pi + 1]:
                        break
                    cand = best_row[:]
                    cand[pi] = new_val
                    # Check internal validity
                    valid = True
                    diffs = set()
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = cand[j] - cand[i]
                            if d in diffs:
                                valid = False
                                break
                            diffs.add(d)
                    if not valid:
                        continue
                    # Check against other rows' differences
                    if diffs & other_set:
                        continue
                    new_max = max(cand)
                    if new_max < best_max:
                        best_row = cand
                        best_max = new_max
                        improved = True
                        break
                    elif new_max == best_max:
                        if random.random() < 0.3:
                            best_row = cand
                            improved = True
                            break
            if not improved:
                break
        return best_row

    def simulated_annealing(rows, n_rows, k, start_time, deadline):
        """SA-based local search with periodic row-level re-optimization."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]

        best_rows = [r[:] for r in current_rows]
        best_val = current_val

        T = 500.0
        T_min = 0.5
        cool = 0.99998

        no_improve = 0
        max_no_improve = 30000
        restarts = 0
        max_restarts = 20
        last_reopt_time = time.time()
        reopt_interval = 1.0

        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]

        def _other_diffs_except(ri):
            other = set()
            for j in range(n_rows):
                if j != ri:
                    other |= row_diffs[j]
            return other

        while time.time() < deadline and no_improve < max_no_improve:
            # Periodic row-level re-optimization
            now = time.time()
            if now - last_reopt_time > reopt_interval and no_improve > 200:
                last_reopt_time = now
                # Use priority queue: pick row with highest "potential" (largest max)
                target_row = max(range(n_rows), key=lambda i: row_maxes[i])
                other_diffs = _other_diffs_except(target_row)
                improved_row = _reoptimize_row(current_rows[target_row], other_diffs, k, max_scope=best_val)
                if max(improved_row) < row_maxes[target_row]:
                    new_diffs = _row_diffs(improved_row)
                    all_used -= row_diffs[target_row]
                    all_used |= new_diffs
                    current_rows[target_row] = improved_row
                    row_diffs[target_row] = new_diffs
                    row_maxes[target_row] = max(improved_row)
                    current_val = max(row_maxes)
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    continue

            # Bias towards rows with the maximum value
            max_r = _max_rows()
            if max_r and random.random() < 0.6:
                ri = random.choice(max_r)
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)

            old_val = current_rows[ri][pi]

            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(80, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)

            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
            max_increase = min(30, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)

            if random.random() < 0.3:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
                if low <= high:
                    candidates.append(random.randint(low, high))

            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue

            candidates.sort(key=lambda v: (v > old_val, v))

            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue

                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val

                new_diffs = set()
                valid = True
                n = len(cand_row)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = cand_row[j] - cand_row[i]
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)

                if not valid:
                    continue

                for d in new_diffs:
                    if d in all_used and d not in row_diffs[ri]:
                        valid = False
                        break

                if not valid:
                    continue

                cand_row_max = max(cand_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max

                delta = new_global_max - current_val

                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.2:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True

                if accept:
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)

                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1

                    moved = True
                    break

            if not moved:
                no_improve += 1

            T = max(T_min, T * cool)

            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                current_rows = [r[:] for r in best_rows]
                current_val = best_val
                all_used = _all_used(current_rows)
                row_diffs = [_row_diffs(r) for r in current_rows]
                row_maxes = [max(r) for r in current_rows]
                T = 300.0
                no_improve = 0

        return best_rows, best_val

    def _row_reopt_greedy(row, all_used_excl, k, max_scope=None):
        """
        Re-optimize a single row using greedy with limited backtracking.
        Returns the optimized row or None.
        """
        if max_scope is None:
            max_scope = 200

        row = [0]
        used_local = set()

        for pos in range(1, k + 1):
            found = False
            start = row[pos - 1] + 1
            end = max_scope + 1

            for v in range(start, end):
                new_diffs = {v - row[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in all_used_excl for d in new_diffs):
                    continue
                if any(d in used_local for d in new_diffs):
                    continue

                row.append(v)
                used_local |= new_diffs
                found = True
                break

            if not found:
                return None

        return row

    def row_reopt_polish(rows, n_rows, k, deadline):
        """
        Polish by iteratively re-optimizing rows using a priority-queue approach:
        always process the row with the highest current max first (highest potential),
        dynamically updating conflicts.
        """
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Priority order: process rows with largest max first
            order = sorted(range(n_rows), key=lambda i: -max(current_rows[i]))
            for ri in order:
                if time.time() > deadline:
                    break

                all_used_excl = set()
                for rj in range(n_rows):
                    if rj != ri:
                        all_used_excl |= row_diffs[rj]

                # Try greedy re-optimization first (fast)
                new_row = _row_reopt_greedy(current_rows[ri], all_used_excl, k, max_scope=current_val)

                if new_row is None:
                    # Fall back to local-search re-optimization
                    new_row = _reoptimize_row(current_rows[ri], all_used_excl, k, max_scope=current_val)

                if new_row is not None:
                    new_row_max = max(new_row)
                    old_row_max = max(current_rows[ri])

                    if new_row_max < old_row_max:
                        all_used -= row_diffs[ri]
                        new_diffs = _row_diffs(new_row)
                        all_used |= new_diffs
                        current_rows[ri] = new_row
                        row_diffs[ri] = new_diffs
                        current_val = _global_max(current_rows)
                        improved = True

        return current_rows, current_val

    # Run multiple SA trials with different seeds
    for trial in range(10):
        if time.time() > deadline - 5:
            break

        seed = trial * 7919 + 12345
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)

        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]

        trial_deadline = min(deadline - 5, start_time + 140.0)
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline)

        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows

    # Final polish with remaining time
    if time.time() < deadline - 2:
        polish_deadline = min(deadline, start_time + 143.0)

        # SA polish
        polish_rows, polish_scope = simulated_annealing(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows

        # Row re-optimization polish with priority queue
        if time.time() < deadline - 1:
            reopt_rows, reopt_scope = row_reopt_polish(
                [r[:] for r in best_solution], n_rows, k, deadline - 1
            )
            if reopt_scope < best_scope:
                best_scope = reopt_scope
                best_solution = reopt_rows

        # Final tight SA pass with very low temperature
        if time.time() < deadline - 0.5:
            final_deadline = deadline - 0.5
            final_rows, final_scope = simulated_annealing(
                [r[:] for r in best_solution], n_rows, k, start_time, final_deadline
            )
            if final_scope < best_scope:
                best_scope = final_scope
                best_solution = final_rows

    return best_solution
# EVOLVE-BLOCK-END
