import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

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

    def greedy_construct(scope_limit=None):
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

    def build_used_array(rows, max_diff):
        used = [False] * (max_diff + 1)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_diff:
                        used[d] = True
        return used

    def simulated_annealing_enhanced(initial_rows, time_limit):
        """
        Enhanced simulated annealing with:
        - Multi-position moves (change 2-3 positions in a row simultaneously)
        - Row swap moves
        - Better cooling schedule
        - Periodic restarts from best if stuck
        """
        rows = [r[:] for r in initial_rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        max_diff = best_scope + 100
        used = build_used_array(rows, max_diff)

        t_end = time.time() + time_limit
        initial_temp = max(2.0, current_scope / 4.0)
        cooling_rate = 0.9997
        temp = initial_temp
        steps = 0
        no_improve = 0
        max_no_improve = 5000

        while time.time() < t_end:
            # Occasionally restart from best if stuck
            if no_improve > max_no_improve and random.random() < 0.1:
                rows = [r[:] for r in best_rows]
                current_scope = best_scope
                used = build_used_array(rows, max_diff)
                temp = initial_temp * 0.5
                no_improve = 0

            move_type = random.random()
            accepted = False

            if move_type < 0.6:
                # Single position move
                row_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                old_val = rows[row_idx][pos]
                min_val = rows[row_idx][pos - 1] + 1
                max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 30

                if min_val <= max_val:
                    for attempt in range(5):
                        new_val = random.randint(min_val, max_val)
                        if new_val == old_val:
                            continue

                        old_diffs = []
                        for i in range(k + 1):
                            if i != pos:
                                d = abs(old_val - rows[row_idx][i])
                                old_diffs.append(d)

                        for d in old_diffs:
                            if d <= max_diff:
                                used[d] = False

                        new_diffs = []
                        valid = True
                        for i in range(k + 1):
                            if i != pos:
                                d = abs(new_val - rows[row_idx][i])
                                if d > max_diff or used[d]:
                                    valid = False
                                    break
                                new_diffs.append(d)

                        if not valid:
                            for d in old_diffs:
                                if d <= max_diff:
                                    used[d] = True
                            continue

                        new_row_max = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
                        other_max = 0
                        for i in range(n_rows):
                            if i != row_idx:
                                if rows[i][-1] > other_max:
                                    other_max = rows[i][-1]
                        new_scope = max(other_max, new_row_max)

                        if new_scope < current_scope or random.random() < math.exp((current_scope - new_scope) / temp):
                            rows[row_idx][pos] = new_val
                            for d in new_diffs:
                                used[d] = True
                            current_scope = new_scope
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in rows]
                                no_improve = 0
                            else:
                                no_improve += 1
                            accepted = True
                            break
                        else:
                            for d in old_diffs:
                                if d <= max_diff:
                                    used[d] = True

            elif move_type < 0.8:
                # Two-position move in same row
                row_idx = random.randint(0, n_rows - 1)
                pos1 = random.randint(1, k)
                pos2 = random.randint(1, k)
                if pos1 == pos2:
                    continue
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1

                old_val1 = rows[row_idx][pos1]
                old_val2 = rows[row_idx][pos2]

                min_val1 = rows[row_idx][pos1 - 1] + 1
                max_val1 = rows[row_idx][pos1 + 1] - 1 if pos1 < k else best_scope + 30
                min_val2 = rows[row_idx][pos2 - 1] + 1
                max_val2 = rows[row_idx][pos2 + 1] - 1 if pos2 < k else best_scope + 30

                # Ensure ordering is maintained
                if pos2 > pos1:
                    min_val2 = max(min_val2, old_val1 + 1)
                if pos1 > pos2:
                    min_val1 = max(min_val1, old_val2 + 1)

                if min_val1 <= max_val1 and min_val2 <= max_val2:
                    for attempt in range(3):
                        new_val1 = random.randint(min_val1, max_val1)
                        new_val2 = random.randint(min_val2, max_val2)
                        if new_val1 == old_val1 and new_val2 == old_val2:
                            continue

                        # Ensure strict ordering
                        if pos1 < pos2:
                            if new_val1 >= new_val2:
                                continue
                        else:
                            if new_val2 >= new_val1:
                                continue

                        # Collect all old diffs
                        old_diffs = set()
                        for i in range(k + 1):
                            if i != pos1 and i != pos2:
                                old_diffs.add(abs(old_val1 - rows[row_idx][i]))
                                old_diffs.add(abs(old_val2 - rows[row_idx][i]))
                        old_diffs.add(abs(old_val1 - old_val2))

                        for d in old_diffs:
                            if d <= max_diff:
                                used[d] = False

                        # Check new diffs
                        new_diffs = []
                        valid = True
                        for i in range(k + 1):
                            if i != pos1 and i != pos2:
                                d1 = abs(new_val1 - rows[row_idx][i])
                                d2 = abs(new_val2 - rows[row_idx][i])
                                if d1 > max_diff or d2 > max_diff or used[d1] or used[d2]:
                                    valid = False
                                    break
                                new_diffs.append(d1)
                                new_diffs.append(d2)
                        d12 = abs(new_val1 - new_val2)
                        if d12 > max_diff or used[d12]:
                            valid = False
                        else:
                            new_diffs.append(d12)

                        if not valid:
                            for d in old_diffs:
                                if d <= max_diff:
                                    used[d] = True
                            continue

                        # Check for duplicate new diffs
                        if len(set(new_diffs)) != len(new_diffs):
                            for d in old_diffs:
                                if d <= max_diff:
                                    used[d] = True
                            continue

                        new_row = rows[row_idx][:]
                        new_row[pos1] = new_val1
                        new_row[pos2] = new_val2
                        new_row_max = max(new_row)
                        other_max = 0
                        for i in range(n_rows):
                            if i != row_idx:
                                if rows[i][-1] > other_max:
                                    other_max = rows[i][-1]
                        new_scope = max(other_max, new_row_max)

                        if new_scope < current_scope or random.random() < math.exp((current_scope - new_scope) / temp):
                            rows[row_idx][pos1] = new_val1
                            rows[row_idx][pos2] = new_val2
                            for d in new_diffs:
                                used[d] = True
                            current_scope = new_scope
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in rows]
                                no_improve = 0
                            else:
                                no_improve += 1
                            accepted = True
                            break
                        else:
                            for d in old_diffs:
                                if d <= max_diff:
                                    used[d] = True

            else:
                # Row swap move
                if n_rows >= 2:
                    r1, r2 = random.sample(range(n_rows), 2)
                    # Swap the rows
                    rows[r1], rows[r2] = rows[r2], rows[r1]
                    # This doesn't change the set of differences, so scope stays same
                    # But it might help escape local minima in subsequent moves
                    # Actually, swapping rows doesn't change anything for validity
                    # Let's do a different move: reverse a row's internal structure
                    # Actually, let's just do a small perturbation
                    rows[r1], rows[r2] = rows[r2], rows[r1]  # swap back
                    # Instead, do a "shift" move: try to shift all elements of a row by -1
                    row_idx = random.randint(0, n_rows - 1)
                    row = rows[row_idx]
                    # Try to decrease the last element by 1
                    old_val = row[k]
                    new_val = old_val - 1
                    if new_val > row[k-1]:
                        old_diffs = [old_val - row[i] for i in range(k)]
                        for d in old_diffs:
                            if d <= max_diff:
                                used[d] = False
                        new_diffs = []
                        valid = True
                        for i in range(k):
                            d = new_val - row[i]
                            if d > max_diff or used[d]:
                                valid = False
                                break
                            new_diffs.append(d)
                        if valid:
                            row[k] = new_val
                            for d in new_diffs:
                                used[d] = True
                            current_scope = get_scope(rows)
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in rows]
                                no_improve = 0
                            else:
                                no_improve += 1
                            accepted = True
                        else:
                            for d in old_diffs:
                                if d <= max_diff:
                                    used[d] = True

            if not accepted:
                no_improve += 1

            temp *= cooling_rate
            if temp < 0.1:
                temp = 0.1
            steps += 1

        return best_rows, best_scope

    def local_search_polish(rows, time_limit):
        """
        Aggressive local search: try to reduce scope by 1 repeatedly.
        For each row, try to decrease each position (from right to left) by 1.
        """
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        t_end = time.time() + time_limit

        max_diff = best_scope + 100
        used = build_used_array(rows, max_diff)

        while time.time() < t_end:
            improved = False
            for row_idx in range(n_rows):
                row = rows[row_idx]
                # Try to decrease positions from right to left
                for pos in range(k, 0, -1):
                    old_val = row[pos]
                    new_val = old_val - 1
                    if new_val <= row[pos - 1]:
                        continue

                    old_diffs = []
                    for i in range(k + 1):
                        if i != pos:
                            d = abs(old_val - row[i])
                            old_diffs.append(d)

                    for d in old_diffs:
                        if d <= max_diff:
                            used[d] = False

                    new_diffs = []
                    valid = True
                    for i in range(k + 1):
                        if i != pos:
                            d = abs(new_val - row[i])
                            if d > max_diff or used[d]:
                                valid = False
                                break
                            new_diffs.append(d)

                    if valid:
                        row[pos] = new_val
                        for d in new_diffs:
                            used[d] = True
                        current_scope = get_scope(rows)
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                        improved = True
                        break  # Restart from beginning after improvement
                    else:
                        for d in old_diffs:
                            if d <= max_diff:
                                used[d] = True

            if not improved:
                # Try more aggressive: allow temporary increases to find better solutions
                # Do a mini-optimization: try random small changes
                for _ in range(100):
                    row_idx = random.randint(0, n_rows - 1)
                    pos = random.randint(1, k)
                    old_val = rows[row_idx][pos]
                    min_val = rows[row_idx][pos - 1] + 1
                    max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 10

                    if min_val > max_val:
                        continue

                    new_val = random.randint(min_val, max_val)
                    if new_val == old_val:
                        continue

                    old_diffs = []
                    for i in range(k + 1):
                        if i != pos:
                            d = abs(old_val - rows[row_idx][i])
                            old_diffs.append(d)

                    for d in old_diffs:
                        if d <= max_diff:
                            used[d] = False

                    new_diffs = []
                    valid = True
                    for i in range(k + 1):
                        if i != pos:
                            d = abs(new_val - rows[row_idx][i])
                            if d > max_diff or used[d]:
                                valid = False
                                break
                            new_diffs.append(d)

                    if valid:
                        # Only accept if it doesn't increase scope
                        new_row_max = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
                        other_max = 0
                        for i in range(n_rows):
                            if i != row_idx:
                                if rows[i][-1] > other_max:
                                    other_max = rows[i][-1]
                        new_scope = max(other_max, new_row_max)
                        if new_scope <= current_scope:
                            rows[row_idx][pos] = new_val
                            for d in new_diffs:
                                used[d] = True
                            current_scope = new_scope
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in rows]
                            improved = True
                            break
                        else:
                            for d in old_diffs:
                                if d <= max_diff:
                                    used[d] = True
                    else:
                        for d in old_diffs:
                            if d <= max_diff:
                                used[d] = True

                if not improved:
                    break

        return best_rows, best_scope

    # Step 1: Greedy construction with multiple tries
    rows = None
    for scope_limit in [112, 120, 130, 140, 150, 160, 200, 300, 500, 1000]:
        rows = greedy_construct(scope_limit)
        if rows is not None:
            break

    if rows is None:
        rows = greedy_construct()
        if rows is None:
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

    current_scope = get_scope(rows)

    # Step 2: Multiple SA runs with different random seeds
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        # Split time among 3 SA runs
        sa_time = (remaining_time - 20) / 3
        best_overall = rows
        best_overall_scope = current_scope

        for sa_run in range(3):
            if time.time() > deadline - 20:
                break
            # Slightly perturb the initial rows for diversity
            perturbed = [r[:] for r in best_overall]
            for _ in range(5):
                ri = random.randint(0, n_rows - 1)
                pi = random.randint(1, k)
                if pi < k:
                    if perturbed[ri][pi] < perturbed[ri][pi+1] - 1:
                        perturbed[ri][pi] += 1
                    elif perturbed[ri][pi] > perturbed[ri][pi-1] + 1:
                        perturbed[ri][pi] -= 1

            if is_valid(perturbed):
                sa_rows, sa_scope = simulated_annealing_enhanced(perturbed, sa_time)
                if is_valid(sa_rows) and sa_scope < best_overall_scope:
                    best_overall = sa_rows
                    best_overall_scope = sa_scope

        rows = best_overall
        current_scope = best_overall_scope

    # Step 3: Final polish
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        polished_rows, polished_scope = local_search_polish(rows, remaining_time - 3)
        if is_valid(polished_rows):
            rows = polished_rows
            current_scope = polished_scope

    return rows
