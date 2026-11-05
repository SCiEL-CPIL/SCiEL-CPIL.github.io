import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s kill (150s total)

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
                for x in range(last_val + 1, min(last_val + 80, scope_limit + 1)):
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
                        if max_d == x - last_val:
                            break
                if best_x is None:
                    # Fallback to linear scan
                    for x in range(last_val + 1, scope_limit + 1):
                        valid = True
                        for y in row:
                            d = x - y
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            best_x = x
                            break
                if best_x is None:
                    return None
                row.append(best_x)
                for y in row[:-1]:
                    used_diffs.add(best_x - y)
            rows.append(row)
        return rows

    def backtracking_row(used_diffs, k, max_scope, rng, time_limit):
        """Build a single row with entries <= max_scope using backtracking."""
        row = [0]
        
        def bt():
            if time.time() > time_limit:
                return False
            if len(row) == k + 1:
                return True
            last_val = row[-1]
            start = last_val + 1
            end = min(max_scope, last_val + 60)
            candidates = list(range(start, end + 1))
            if rng and len(candidates) > 12:
                rng.shuffle(candidates[6:])
            for x in candidates:
                if time.time() > time_limit:
                    return False
                new_diffs = set()
                valid = True
                for y in row:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                # Forward check
                remaining = k + 1 - len(row) - 1
                if remaining > 0 and x + remaining > max_scope:
                    continue
                
                row.append(x)
                used_diffs.update(new_diffs)
                if bt():
                    return True
                row.pop()
                used_diffs.difference_update(new_diffs)
            return False
        
        if bt():
            return row
        return None

    def rebuild_row(used_other, k, max_scope, rng, time_limit):
        """Rebuild a single row given differences used by others."""
        return backtracking_row(used_other, k, max_scope, rng, time_limit)

    def all_diffs_except(rows, skip_idx):
        s = set()
        for idx, row in enumerate(rows):
            if idx == skip_idx:
                continue
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    def local_search(rows, time_budget, rng=None):
        """Local search: rebuild the worst row to reduce scope."""
        if rng is None:
            rng = random.Random(42)
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        t_end = time.time() + time_budget
        no_improve = 0
        max_no_improve = 500

        while time.time() < t_end and no_improve < max_no_improve:
            # Find row with largest max entry
            max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            current_max = rows[max_idx][-1]
            target = best_scope - 1
            
            if target < k + 1:
                break

            used_other = all_diffs_except(rows, max_idx)
            time_limit = time.time() + 0.5  # 500ms per rebuild attempt
            
            new_row = rebuild_row(used_other, k, target, rng, time_limit)
            if new_row is not None:
                rows[max_idx] = new_row
                new_scope = get_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        return best_rows, best_scope

    def vns_search(rows, time_budget, rng=None):
        """
        Variable Neighborhood Search: perturb by swapping difference assignments
        between rows, then rebuild affected rows.
        """
        if rng is None:
            rng = random.Random(42)
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        t_end = time.time() + time_budget
        no_improve = 0
        max_no_improve = 300
        neighborhood = 1

        while time.time() < t_end and no_improve < max_no_improve:
            # Perturb: swap some elements between two rows
            r1, r2 = rng.sample(range(n_rows), 2)
            # Pick positions to swap (not position 0)
            p1 = rng.randint(1, k)
            p2 = rng.randint(1, k)
            
            old_v1 = rows[r1][p1]
            old_v2 = rows[r2][p2]
            
            # Try the swap
            rows[r1][p1] = old_v2
            rows[r2][p2] = old_v1
            
            # Check validity
            if is_valid(rows):
                new_scope = get_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                    neighborhood = 1
                else:
                    no_improve += 1
                    neighborhood += 1
            else:
                # Revert
                rows[r1][p1] = old_v1
                rows[r2][p2] = old_v2
                no_improve += 1
                neighborhood += 1

            # If no improvement, increase perturbation strength
            if no_improve % 50 == 0:
                # Try rebuilding a random row
                idx = rng.randint(0, n_rows - 1)
                used_other = all_diffs_except(rows, idx)
                target = best_scope - 1
                if target >= k + 1:
                    time_limit = time.time() + 0.3
                    new_row = rebuild_row(used_other, k, target, rng, time_limit)
                    if new_row is not None:
                        rows[idx] = new_row
                        if is_valid(rows):
                            new_scope = get_scope(rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [r[:] for r in rows]
                                no_improve = 0
                                neighborhood = 1

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

            if pos == k:
                new_val = rng.randint(min_val, max(min_val + 20, max_val))
            else:
                new_val = rng.randint(min_val, max_val)

            if new_val == old_val:
                current_temp *= cooling_rate
                continue

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

            if pos == k:
                new_row_max = new_val
            else:
                new_row_max = max(rows[row_idx])

            other_max = 0
            for i in range(n_rows):
                if i != row_idx:
                    other_max = max(other_max, max(rows[i]))

            new_scope = max(new_row_max, other_max)
            delta = current_scope - new_scope

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

    # === Main strategy ===

    # Step 1: Get initial solution using multiple greedy strategies
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
        initial_rows = greedy_construct(2000, random.Random(42))
        if initial_rows is None:
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

    # Step 2: Local search (rebuild worst row)
    remaining = deadline - time.time()
    if remaining > 20:
        ls_rows, ls_scope = local_search(initial_rows, remaining * 0.3, random.Random(42))
        if ls_rows is not None and is_valid(ls_rows) and ls_scope < current_scope:
            initial_rows = ls_rows
            current_scope = ls_scope

    # Step 3: VNS (swap-based perturbation + rebuild)
    remaining = deadline - time.time()
    if remaining > 20:
        vns_rows, vns_scope = vns_search(initial_rows, remaining * 0.25, random.Random(123))
        if vns_rows is not None and is_valid(vns_rows) and vns_scope < current_scope:
            initial_rows = vns_rows
            current_scope = vns_scope

    # Step 4: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 20:
        sa_rows, sa_scope = simulated_annealing(initial_rows, remaining * 0.25, random.Random(777))
        if sa_rows is not None and is_valid(sa_rows) and sa_scope < current_scope:
            initial_rows = sa_rows
            current_scope = sa_scope

    # Step 5: Final local search polish
    remaining = deadline - time.time()
    if remaining > 15:
        final_rows, final_scope = local_search(initial_rows, remaining * 0.6, random.Random(999))
        if final_rows is not None and is_valid(final_rows) and final_scope < current_scope:
            initial_rows = final_rows
            current_scope = final_scope

    # Step 6: One more VNS pass if time permits
    remaining = deadline - time.time()
    if remaining > 10:
        vns2_rows, vns2_scope = vns_search(initial_rows, remaining * 0.5, random.Random(555))
        if vns2_rows is not None and is_valid(vns2_rows) and vns2_scope < current_scope:
            initial_rows = vns2_rows
            current_scope = vns2_scope

    return initial_rows
