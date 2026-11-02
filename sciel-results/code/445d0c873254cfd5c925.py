import time
import random
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses multi-trial simulated annealing with fast incremental validity checks,
    row-level re-optimization, and a right-to-left greedy polish.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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

    def _validate(rows, n_rows, k):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

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
        """Greedy initialization with randomization."""
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

    def _row_reopt(row, all_used_excl, k, max_scope=None):
        """
        Re-optimize a single row given the differences used by all other rows.
        Greedy fill with limited backtracking.
        """
        if max_scope is None:
            max_scope = 200

        row_greedy = [0]
        used_local = set()
        for pos in range(1, k + 1):
            found = False
            start = row_greedy[pos - 1] + 1
            end = max_scope + 1

            for v in range(start, end):
                new_diffs = {v - row_greedy[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in all_used_excl for d in new_diffs):
                    continue
                if any(d in used_local for d in new_diffs):
                    continue

                row_greedy.append(v)
                used_local |= new_diffs
                found = True
                break

            if not found:
                return None

        # Limited backtracking refinement
        def _try_fill_bt(row, pos, used_local, depth, max_depth=2):
            if pos == k + 1:
                return row[:]
            start = row[pos - 1] + 1
            for v in range(start, max_scope + 1):
                new_diffs = {v - row[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in all_used_excl for d in new_diffs):
                    continue
                if any(d in used_local for d in new_diffs):
                    continue
                row[pos] = v
                used_local_new = used_local | new_diffs
                if depth < max_depth:
                    result = _try_fill_bt(row, pos + 1, used_local_new, depth + 1, max_depth)
                    if result is not None:
                        return result
                else:
                    rest = row[:]
                    rest_used = used_local_new
                    ok = True
                    for p in range(pos + 1, k + 1):
                        found = False
                        for v2 in range(rest[p-1] + 1, max_scope + 1):
                            nd2 = {v2 - rest[i] for i in range(p)}
                            if len(nd2) != p:
                                continue
                            if any(d in all_used_excl for d in nd2):
                                continue
                            if any(d in rest_used for d in nd2):
                                continue
                            rest.append(v2)
                            rest_used |= nd2
                            found = True
                            break
                        if not found:
                            ok = False
                            break
                    if ok:
                        return rest
            return None

        bt_row = [0] + [0] * k
        bt_result = _try_fill_bt(bt_row, 1, set(), 0)
        if bt_result is not None and max(bt_result) < max(row_greedy):
            return bt_result
        return row_greedy

    def row_reopt_polish(rows, n_rows, k, deadline):
        """Polish by re-optimizing each row against the others."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri in range(n_rows):
                all_used_excl = set()
                for rj in range(n_rows):
                    if rj != ri:
                        all_used_excl |= row_diffs[rj]

                new_row = _row_reopt(current_rows[ri], all_used_excl, k, max_scope=current_val - 1)

                if new_row is not None:
                    new_row_max = max(new_row)
                    old_row_max = max(current_rows[ri])

                    if new_row_max < old_row_max:
                        current_rows[ri] = new_row
                        row_diffs[ri] = _row_diffs(new_row)
                        current_val = _global_max(current_rows)
                        improved = True

        return current_rows, current_val

    def simulated_annealing(rows, n_rows, k, start_time, deadline):
        """SA-based local search with fast incremental validity checks."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]

        best_rows = [r[:] for r in current_rows]
        best_val = current_val

        T = 300.0
        T_min = 0.1
        cool = 0.99995

        no_improve = 0
        max_no_improve = 15000
        restarts = 0
        max_restarts = 30
        last_reopt_time = time.time()
        reopt_interval = 0.8

        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]

        def _other_diffs_except(ri):
            other = set()
            for j in range(n_rows):
                if j != ri:
                    other |= row_diffs[j]
            return other

        while time.time() < deadline and no_improve < max_no_improve:
            now = time.time()
            if now - last_reopt_time > reopt_interval and no_improve > 100:
                last_reopt_time = now
                target_row = max(range(n_rows), key=lambda i: row_maxes[i])
                other_diffs = _other_diffs_except(target_row)
                improved_row = _row_reopt(current_rows[target_row], other_diffs, k, max_scope=best_val - 1)
                if improved_row is not None and max(improved_row) < row_maxes[target_row]:
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

            max_r = _max_rows()
            if max_r and random.random() < 0.7:
                ri = random.choice(max_r)
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)

            old_val = current_rows[ri][pi]
            row = current_rows[ri]

            # Compute other_row_diffs once (diffs not involving position pi)
            old_involving = set()
            for i in range(k + 1):
                if i == pi:
                    continue
                d = old_val - row[i] if i < pi else row[i] - old_val
                old_involving.add(d)
            other_row_diffs = row_diffs[ri] - old_involving

            candidates = []
            min_allowed = row[pi - 1] + 1
            max_decrease = min(120, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)

            max_allowed = (row[pi + 1] - 1) if pi < k else best_val + 30
            max_increase = min(40, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)

            if random.random() < 0.3:
                low = row[pi - 1] + 1
                high = (row[pi + 1] - 1) if pi < k else best_val + 20
                if low <= high:
                    candidates.append(random.randint(low, high))

            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue

            candidates.sort(key=lambda v: (v > old_val, v))

            moved = False
            for new_val in candidates:
                if new_val < 0 or new_val <= row[pi - 1]:
                    continue
                if pi < k and new_val >= row[pi + 1]:
                    continue
                if new_val == old_val:
                    continue

                # Fast incremental validity check:
                new_diffs = set()
                valid = True
                for i in range(k + 1):
                    if i == pi:
                        continue
                    d = new_val - row[i] if i < pi else row[i] - new_val
                    if d in new_diffs or d in other_row_diffs:
                        valid = False
                        break
                    new_diffs.add(d)

                if not valid:
                    continue

                # Check against all_used excluding this row's current diffs
                for d in new_diffs:
                    if d in all_used and d not in row_diffs[ri]:
                        valid = False
                        break

                if not valid:
                    continue

                # Compute new row max efficiently
                if pi == k:
                    cand_row_max = new_val
                else:
                    cand_row_max = row_maxes[ri]
                    if new_val > cand_row_max:
                        cand_row_max = new_val

                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max

                delta = new_global_max - current_val

                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.4:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True

                if accept:
                    # Update state incrementally
                    old_row_diffs = row_diffs[ri]
                    all_used -= old_row_diffs
                    all_used |= new_diffs
                    all_used |= other_row_diffs

                    row[pi] = new_val
                    row_diffs[ri] = new_diffs | other_row_diffs
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
                T = 200.0
                no_improve = 0

        return best_rows, best_val

    def greedy_polish_right_to_left(rows, n_rows, k, deadline):
        """
        Right-to-left greedy polish: for each row, try to reduce each position
        from right to left to the smallest valid value.
        """
        best_rows = [r[:] for r in rows]
        used_diffs = _all_used(best_rows)

        while time.time() < deadline - 2:
            progress = False
            row_order = list(range(n_rows))
            random.shuffle(row_order)

            for r_idx in row_order:
                if time.time() > deadline - 2:
                    break
                row = best_rows[r_idx]

                for pos in range(k, 0, -1):
                    old_val = row[pos]

                    # Diffs involving old_val at pos
                    diffs_involving_old = set()
                    for i in range(k + 1):
                        if i == pos:
                            continue
                        if i < pos:
                            diffs_involving_old.add(old_val - row[i])
                        else:
                            diffs_involving_old.add(row[i] - old_val)

                    forbidden_diffs = used_diffs - diffs_involving_old

                    lower_bound = row[pos - 1] + 1
                    upper_bound = old_val - 1
                    if pos + 1 < k + 1:
                        upper_bound = min(upper_bound, row[pos + 1] - 1)

                    if upper_bound < lower_bound:
                        continue

                    found = None
                    for new_val in range(lower_bound, upper_bound + 1):
                        if new_val == old_val:
                            continue
                        valid = True
                        new_diffs = set()
                        for i in range(k + 1):
                            if i == pos:
                                continue
                            d = new_val - row[i] if i < pos else row[i] - new_val
                            if d in forbidden_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)

                        if valid:
                            found = new_val
                            break

                    if found is not None and found < old_val:
                        # Update used_diffs
                        used_diffs -= diffs_involving_old
                        used_diffs |= new_diffs
                        row[pos] = found
                        progress = True

            if not progress:
                break

        return best_rows

    # Run multiple SA trials with different seeds
    for trial in range(15):
        if time.time() > deadline - 8:
            break

        seed = trial * 7919 + 12345
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)

        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]

        # Give more time to early trials, less to later ones
        remaining = deadline - 8 - time.time()
        if remaining < 5:
            break
        trial_deadline = time.time() + min(remaining * 0.6, 15.0)
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline)

        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows

    # Final polish with remaining time
    if time.time() < deadline - 3:
        polish_deadline = min(deadline - 1, start_time + 115.0)

        polish_rows, polish_scope = simulated_annealing(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows

        if time.time() < deadline - 1.5:
            reopt_rows, reopt_scope = row_reopt_polish(
                [r[:] for r in best_solution], n_rows, k, deadline - 1.5
            )
            if reopt_scope < best_scope:
                best_scope = reopt_scope
                best_solution = reopt_rows

        # Right-to-left greedy polish
        if time.time() < deadline - 0.5:
            polished = greedy_polish_right_to_left(
                [r[:] for r in best_solution], n_rows, k, deadline - 0.5
            )
            polished_scope = _global_max(polished)
            if polished_scope < best_scope:
                best_scope = polished_scope
                best_solution = polished

        # Final tight SA polish if time remains
        if time.time() < deadline - 0.3:
            final_deadline = deadline - 0.3
            final_rows, final_scope = simulated_annealing(
                [r[:] for r in best_solution], n_rows, k, start_time, final_deadline
            )
            if final_scope < best_scope:
                best_scope = final_scope
                best_solution = final_rows

    return best_solution
# EVOLVE-BLOCK-END
