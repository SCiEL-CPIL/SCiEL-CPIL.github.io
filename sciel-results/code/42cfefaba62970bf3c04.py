import random
import time
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    def get_max_entry(rows):
        return max(max(row) for row in rows)

    def get_row_diffs(row):
        diffs = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diffs |= (1 << d)
        return diffs

    def is_valid_row(row, global_used_except_self):
        row_diffs = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if (row_diffs >> d) & 1:
                    return False, 0
                row_diffs |= (1 << d)
        if row_diffs & global_used_except_self:
            return False, 0
        return True, row_diffs

    def greedy_row(max_val, global_used):
        row = [0]
        current_diffs = 0
        while len(row) < k + 1:
            last = row[-1]
            found = False
            for v in range(last + 1, max_val + 1):
                conflict = False
                new_diffs = 0
                for u in row:
                    d = v - u
                    if (current_diffs >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if conflict:
                    continue
                if new_diffs & global_used:
                    continue
                row.append(v)
                current_diffs |= new_diffs
                found = True
                break
            if not found:
                return None
        return row

    def pure_greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_diffs = 0
            v = 1
            while len(row) < k + 1:
                conflict = False
                new_diffs = 0
                for u in row:
                    d = v - u
                    if (current_diffs >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if not conflict and not (new_diffs & used):
                    row.append(v)
                    current_diffs |= new_diffs
                v += 1
            rows.append(row)
            used |= current_diffs
        return rows

    # Compute cost for SA
    def compute_cost(rows, max_limit):
        used_diffs = {}
        max_entry = 0
        for row in rows:
            if row:
                m = max(row)
                if m > max_entry:
                    max_entry = m
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        used_diffs[d] += 1
                    else:
                        used_diffs[d] = 1
        num_conflicts = 0
        for count in used_diffs.values():
            if count > 1:
                num_conflicts += count - 1
        conflict_penalty = 10000 * num_conflicts
        limit_penalty = 0
        if max_entry > max_limit:
            limit_penalty = 1000 * (max_entry - max_limit)
        return max_entry + conflict_penalty + limit_penalty

    def is_valid_solution(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # Initial baseline using pure greedy
    best_rows = pure_greedy()
    best_max = get_max_entry(best_rows)

    # SA Parameters
    T_start = 100.0
    T_end = 1.0
    cooling_rate = 0.999
    num_restarts = 20

    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        # Initialize SA with random configuration
        init_max = best_max + 10
        current_rows = []
        for _ in range(n_rows):
            if init_max < k:
                row = list(range(k + 1))
            else:
                vals = random.sample(range(1, init_max + 1), k)
                vals.sort()
                row = [0] + vals
            current_rows.append(row)

        T = T_start
        current_cost = compute_cost(current_rows, best_max - 1)

        # Check if initial is already good
        if is_valid_solution(current_rows):
            curr_max = get_max_entry(current_rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = [row[:] for row in current_rows]

        while T > T_end and time.time() < deadline - 5:
            # Generate a neighbor
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)

            min_val = current_rows[r_idx][c_idx - 1] + 1
            max_val = best_max - 1

            if c_idx < k:
                max_val = min(max_val, current_rows[r_idx][c_idx + 1] - 1)

            if min_val > max_val:
                T *= cooling_rate
                continue

            new_val = random.randint(min_val, max_val)

            if new_val == current_rows[r_idx][c_idx]:
                T *= cooling_rate
                continue

            old_val = current_rows[r_idx][c_idx]
            current_rows[r_idx][c_idx] = new_val

            new_cost = compute_cost(current_rows, best_max - 1)
            delta_cost = new_cost - current_cost

            if delta_cost < 0 or (T > 0 and random.random() < math.exp(-delta_cost / T)):
                current_cost = new_cost

                # Check if we found a valid solution better than best
                if new_cost < best_max:
                    if is_valid_solution(current_rows):
                        curr_max = get_max_entry(current_rows)
                        if curr_max < best_max:
                            best_max = curr_max
                            best_rows = [row[:] for row in current_rows]

            else:
                # Revert
                current_rows[r_idx][c_idx] = old_val

            T *= cooling_rate

    # Local search to polish
    no_improve_count = 0
    while time.time() < deadline - 2.0:
        row_idx = random.randint(0, n_rows - 1)

        global_used_except = 0
        for i in range(n_rows):
            if i != row_idx:
                global_used_except |= get_row_diffs(best_rows[i])

        curr_row_max = best_rows[row_idx][-1]
        target = curr_row_max - 1

        new_row = None
        for _ in range(20):
            row = [0]
            current_diffs = 0
            success = True
            for _ in range(k):
                last = row[-1]
                candidates = list(range(last + 1, target + 1))
                random.shuffle(candidates)
                found = False
                for v in candidates:
                    conflict = False
                    new_diffs = 0
                    for u in row:
                        d = v - u
                        if (current_diffs >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    if conflict:
                        continue
                    if new_diffs & global_used_except:
                        continue
                    row.append(v)
                    current_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    success = False
                    break
            if success:
                valid, _ = is_valid_row(row, global_used_except)
                if valid:
                    new_row = row
                    break

        if new_row is not None:
            best_rows[row_idx] = new_row
            new_global_max = get_max_entry(best_rows)
            if new_global_max < best_max:
                best_max = new_global_max
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        if no_improve_count > 100:
            # Perturbation
            pert_idx = random.randint(0, n_rows - 1)
            pert_scope = best_max + 10
            global_used_except_pert = 0
            for i in range(n_rows):
                if i != pert_idx:
                    global_used_except_pert |= get_row_diffs(best_rows[i])

            new_pert_row = None
            for _ in range(10):
                row = [0]
                current_diffs = 0
                success = True
                for _ in range(k):
                    last = row[-1]
                    candidates = list(range(last + 1, pert_scope + 1))
                    random.shuffle(candidates)
                    found = False
                    for v in candidates:
                        conflict = False
                        new_diffs = 0
                        for u in row:
                            d = v - u
                            if (current_diffs >> d) & 1:
                                conflict = True
                                break
                            new_diffs |= (1 << d)
                        if conflict:
                            continue
                        if new_diffs & global_used_except_pert:
                            continue
                        row.append(v)
                        current_diffs |= new_diffs
                        found = True
                        break
                    if not found:
                        success = False
                        break
                if success:
                    valid, _ = is_valid_row(row, global_used_except_pert)
                    if valid:
                        new_pert_row = row
                        break

            if new_pert_row is not None:
                best_rows[pert_idx] = new_pert_row
                no_improve_count = 0

    # Final Validation
    if not is_valid_solution(best_rows):
        rows_fallback = pure_greedy()
        best_rows = rows_fallback
        best_max = get_max_entry(best_rows)

    return best_rows
