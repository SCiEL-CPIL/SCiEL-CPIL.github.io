import time
import random
import math


def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    n_diffs_per_row = k * (k + 1) // 2

    def greedy_construct():
        """Fast greedy construction to get a baseline."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = [r[:] for r in baseline_rows]
    best_max = max(max(r) for r in best_rows)

    def compute_cost(rows):
        used = set()
        max_val = 0
        total_sum = 0
        for r in rows:
            if r[-1] > max_val:
                max_val = r[-1]
            total_sum += sum(r)
            for i in range(len(r)):
                ri = r[i]
                for j in range(i + 1, len(r)):
                    d = r[j] - ri
                    if d in used:
                        return 1, max_val, total_sum
                    used.add(d)
        return 0, max_val, total_sum

    def sa_optimize(rows, time_limit):
        """Simulated annealing to minimize the scope."""
        # Build diff count structure
        diff_count = {}
        collisions = 0
        max_val = 0
        total_sum = 0
        for r in rows:
            max_val = max(max_val, r[-1])
            total_sum += sum(r)
            for i in range(len(r)):
                ri = r[i]
                for j in range(i + 1, len(r)):
                    d = r[j] - ri
                    if d in diff_count:
                        diff_count[d] += 1
                        if diff_count[d] == 2:
                            collisions += 1
                    else:
                        diff_count[d] = 1

        current_collisions = collisions
        current_max = max_val
        current_sum = total_sum
        best_collisions = current_collisions
        best_max = current_max
        best_sum = current_sum
        best_rows_copy = [r[:] for r in rows]

        T = 30.0
        cooling = 0.9997
        start = time.time()
        rows_copy = [r[:] for r in rows]

        while time.time() - start < time_limit:
            r_idx = random.randint(0, n_rows - 1)
            # Bias toward tail positions
            if random.random() < 0.5:
                i_idx = random.randint(max(1, row_len - 3), row_len - 1)
            else:
                i_idx = random.randint(1, row_len - 1)

            row = rows_copy[r_idx]
            old_val = row[i_idx]
            lower = row[i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = row[i_idx + 1] - 1
            else:
                upper = best_max + 5

            if lower > upper:
                continue

            # Try to reduce or randomly perturb
            if random.random() < 0.4:
                new_val = lower
            else:
                new_val = random.randint(lower, upper)

            if new_val == old_val:
                continue

            # Remove old differences
            dc = diff_count
            for j in range(row_len):
                if j == i_idx:
                    continue
                d = abs(old_val - row[j])
                c = dc[d] - 1
                if c == 1:
                    collisions -= 1
                if c == 0:
                    del dc[d]
                else:
                    dc[d] = c

            total_sum -= old_val
            row[i_idx] = new_val
            total_sum += new_val

            # Add new differences
            for j in range(row_len):
                if j == i_idx:
                    continue
                d = abs(new_val - row[j])
                if d in dc:
                    c = dc[d] + 1
                    dc[d] = c
                    if c == 2:
                        collisions += 1
                else:
                    dc[d] = 1

            new_max = max(max_val, new_val)
            if old_val == max_val and new_val < old_val:
                new_max = max(max(r) for r in rows_copy)

            # Cost evaluation
            if collisions < current_collisions:
                accept = True
            elif collisions == current_collisions:
                if new_max < current_max:
                    accept = True
                elif new_max == current_max:
                    if total_sum < current_sum:
                        accept = True
                    else:
                        delta = (total_sum - current_sum) * 0.1
                        accept = random.random() < math.exp(-delta / max(T, 0.1))
                else:
                    delta = (new_max - current_max) * 10 + (total_sum - current_sum) * 0.1
                    accept = random.random() < math.exp(-delta / max(T, 0.1))
            else:
                delta = (collisions - current_collisions) * 50 + (new_max - current_max) + (total_sum - current_sum) * 0.01
                accept = random.random() < math.exp(-delta / max(T, 0.1))

            if accept:
                current_collisions = collisions
                current_max = new_max
                current_sum = total_sum
                max_val = new_max
                if current_collisions < best_collisions or \
                   (current_collisions == best_collisions and
                    (current_max < best_max or (current_max == best_max and current_sum < best_sum))):
                    best_collisions = current_collisions
                    best_max = current_max
                    best_sum = current_sum
                    best_rows_copy = [r[:] for r in rows_copy]
            else:
                # Undo
                for j in range(row_len):
                    if j == i_idx:
                        continue
                    d = abs(new_val - row[j])
                    c = dc[d] - 1
                    if c == 1:
                        collisions -= 1
                    if c == 0:
                        del dc[d]
                    else:
                        dc[d] = c

                total_sum -= new_val
                row[i_idx] = old_val
                total_sum += old_val

                for j in range(row_len):
                    if j == i_idx:
                        continue
                    d = abs(old_val - row[j])
                    if d in dc:
                        c = dc[d] + 1
                        dc[d] = c
                        if c == 2:
                            collisions += 1
                    else:
                        dc[d] = 1

                if old_val > max_val:
                    max_val = old_val
                elif new_val == max_val and old_val < new_val:
                    max_val = max(max(r) for r in rows_copy)

            T *= cooling
            if T < 0.3:
                T = 0.3

        return best_rows_copy, best_max, best_sum, best_collisions

    def local_polish(rows, time_limit):
        """Aggressive local search: only accept valid moves that reduce scope."""
        diff_count = {}
        collisions = 0
        max_val = 0
        total_sum = 0
        for r in rows:
            max_val = max(max_val, r[-1])
            total_sum += sum(r)
            for i in range(len(r)):
                ri = r[i]
                for j in range(i + 1, len(r)):
                    d = r[j] - ri
                    if d in diff_count:
                        diff_count[d] += 1
                        if diff_count[d] == 2:
                            collisions += 1
                    else:
                        diff_count[d] = 1

        if collisions > 0:
            return rows, max_val, total_sum, collisions

        best_max = max_val
        best_sum = total_sum
        best_rows_copy = [r[:] for r in rows]
        start = time.time()
        rows_copy = [r[:] for r in rows]

        while time.time() - start < time_limit:
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            row = rows_copy[r_idx]
            old_val = row[i_idx]
            lower = row[i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = row[i_idx + 1] - 1
            else:
                upper = row[i_idx] - 1  # try to reduce

            if lower > upper:
                continue

            # Try tightest feasible value first
            cands = [lower]
            span = upper - lower + 1
            if span > 1:
                for _ in range(3):
                    cands.append(random.randint(lower, upper))

            improved = False
            for new_val in cands:
                if new_val == old_val:
                    continue

                # Remove old differences
                dc = diff_count
                for j in range(row_len):
                    if j == i_idx:
                        continue
                    d = abs(old_val - row[j])
                    c = dc[d] - 1
                    if c == 1:
                        collisions -= 1
                    if c == 0:
                        del dc[d]
                    else:
                        dc[d] = c

                total_sum -= old_val
                row[i_idx] = new_val
                total_sum += new_val

                # Add new differences
                for j in range(row_len):
                    if j == i_idx:
                        continue
                    d = abs(new_val - row[j])
                    if d in dc:
                        c = dc[d] + 1
                        dc[d] = c
                        if c == 2:
                            collisions += 1
                    else:
                        dc[d] = 1

                if collisions == 0:
                    new_max = max(max_val, new_val)
                    if old_val == max_val and new_val < old_val:
                        new_max = max(max(r) for r in rows_copy)
                    if new_max < best_max or (new_max == best_max and total_sum < best_sum):
                        best_max = new_max
                        best_sum = total_sum
                        best_rows_copy = [r[:] for r in rows_copy]
                        max_val = new_max
                        improved = True
                    else:
                        # Undo
                        for j in range(row_len):
                            if j == i_idx:
                                continue
                            d = abs(new_val - row[j])
                            c = dc[d] - 1
                            if c == 1:
                                collisions -= 1
                            if c == 0:
                                del dc[d]
                            else:
                                dc[d] = c
                        total_sum -= new_val
                        row[i_idx] = old_val
                        total_sum += old_val
                        for j in range(row_len):
                            if j == i_idx:
                                continue
                            d = abs(old_val - row[j])
                            if d in dc:
                                c = dc[d] + 1
                                dc[d] = c
                                if c == 2:
                                    collisions += 1
                            else:
                                dc[d] = 1
                        if old_val > max_val:
                            max_val = old_val
                else:
                    # Undo
                    for j in range(row_len):
                        if j == i_idx:
                            continue
                        d = abs(new_val - row[j])
                        c = dc[d] - 1
                        if c == 1:
                            collisions -= 1
                        if c == 0:
                            del dc[d]
                        else:
                            dc[d] = c
                    total_sum -= new_val
                    row[i_idx] = old_val
                    total_sum += old_val
                    for j in range(row_len):
                        if j == i_idx:
                            continue
                        d = abs(old_val - row[j])
                        if d in dc:
                            c = dc[d] + 1
                            dc[d] = c
                            if c == 2:
                                collisions += 1
                        else:
                            dc[d] = 1
                    if old_val > max_val:
                        max_val = old_val
                    break  # collision, stop trying other candidates

            if not improved:
                # Occasionally try a random valid perturbation to escape
                if random.random() < 0.1:
                    r_idx = random.randint(0, n_rows - 1)
                    i_idx = random.randint(1, row_len - 1)
                    row = rows_copy[r_idx]
                    old_val = row[i_idx]
                    lower = row[i_idx - 1] + 1
                    if i_idx < row_len - 1:
                        upper = row[i_idx + 1] - 1
                    else:
                        upper = best_max + 2
                    if lower <= upper:
                        new_val = random.randint(lower, upper)
                        if new_val != old_val:
                            dc = diff_count
                            for j in range(row_len):
                                if j == i_idx:
                                    continue
                                d = abs(old_val - row[j])
                                c = dc[d] - 1
                                if c == 1:
                                    collisions -= 1
                                if c == 0:
                                    del dc[d]
                                else:
                                    dc[d] = c
                            total_sum -= old_val
                            row[i_idx] = new_val
                            total_sum += new_val
                            for j in range(row_len):
                                if j == i_idx:
                                    continue
                                d = abs(new_val - row[j])
                                if d in dc:
                                    c = dc[d] + 1
                                    dc[d] = c
                                    if c == 2:
                                        collisions += 1
                                else:
                                    dc[d] = 1
                            if collisions == 0:
                                new_max = max(max_val, new_val)
                                if old_val == max_val and new_val < old_val:
                                    new_max = max(max(r) for r in rows_copy)
                                max_val = new_max
                            else:
                                # Undo
                                for j in range(row_len):
                                    if j == i_idx:
                                        continue
                                    d = abs(new_val - row[j])
                                    c = dc[d] - 1
                                    if c == 1:
                                        collisions -= 1
                                    if c == 0:
                                        del dc[d]
                                    else:
                                        dc[d] = c
                                total_sum -= new_val
                                row[i_idx] = old_val
                                total_sum += old_val
                                for j in range(row_len):
                                    if j == i_idx:
                                        continue
                                    d = abs(old_val - row[j])
                                    if d in dc:
                                        c = dc[d] + 1
                                        dc[d] = c
                                        if c == 2:
                                            collisions += 1
                                    else:
                                        dc[d] = 1
                                if old_val > max_val:
                                    max_val = old_val

        return best_rows_copy, best_max, best_sum, 0

    # Main loop
    sa_end = start_time + 110.0
    while time.time() < sa_end:
        if random.random() < 0.4:
            init_rows = [r[:] for r in best_rows]
        else:
            bound = best_max + 8
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, bound + 1), k)
                pts.sort()
                init_rows.append([0] + pts)

        sa_time = min(8.0, sa_end - time.time())
        if sa_time < 1.0:
            break

        local_rows, local_max, local_sum, local_collisions = sa_optimize(init_rows, sa_time)

        if local_collisions == 0 and \
           (local_max < best_max or (local_max == best_max and local_sum < sum(sum(r) for r in best_rows))):
            best_rows = local_rows
            best_max = local_max

        # Polish if valid
        if compute_cost(best_rows)[0] == 0 and time.time() < sa_end - 5.0:
            polish_time = min(3.0, sa_end - time.time())
            polished_rows, polished_max, polished_sum, _ = local_polish(best_rows, polish_time)
            if polished_max < best_max or (polished_max == best_max and polished_sum < sum(sum(r) for r in best_rows)):
                best_rows = polished_rows
                best_max = polished_max

    # Final polish
    if compute_cost(best_rows)[0] == 0 and time.time() < deadline - 3.0:
        polish_time = min(30.0, deadline - time.time() - 3.0)
        if polish_time > 1.0:
            polished_rows, polished_max, polished_sum, _ = local_polish(best_rows, polish_time)
            if polished_max < best_max or (polished_max == best_max and polished_sum < sum(sum(r) for r in best_rows)):
                best_rows = polished_rows
                best_max = polished_max

    return best_rows
