import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    start = time.time()
    deadline = start + 118.0
    best_max = float('inf')
    best_rows = None

    def get_diffs(row):
        L = len(row)
        return [row[j] - row[i] for i in range(L) for j in range(i + 1, L)]

    # --- Greedy with lookahead: pick smallest x whose diff-set conflicts least ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                best_x = None
                best_conf = None
                last = row[-1]
                for x in range(last + 1, last + 400):
                    nd = {x - a for a in row}
                    c = len(nd & used)
                    if c == 0:
                        best_x = x
                        best_conf = 0
                        break
                    if best_conf is None or c < best_conf:
                        best_conf = c
                        best_x = x
                        if c <= 1:
                            break
                row.append(best_x)
                used |= {best_x - a for a in row[:-1]}
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = max(max(r) for r in best_rows)

    # --- SA with delta-based conflict tracking (O(1) cost per move) ---
    def sa_optimize(initial_rows, time_limit, bound):
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len

        row_diffs = [get_diffs(r) for r in rows]
        diff_counts = {}
        collision_count = 0
        for rd in row_diffs:
            for d in rd:
                if d in diff_counts:
                    if diff_counts[d] == 1:
                        collision_count += 1
                    diff_counts[d] += 1
                else:
                    diff_counts[d] = 1

        current_max_val = max(r[-1] for r in rows)
        current_cost = collision_count * 10000 + current_max_val
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max_val

        T = 30.0
        cooling = 0.99995
        start_time = time.time()

        while time.time() - start_time < time_limit:
            r_idx = random.randint(0, n - 1)
            i_idx = random.randint(1, L - 1)
            row = rows[r_idx]
            old_val = row[i_idx]

            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < L - 1 else bound + 12
            upper = min(upper, bound + 20)
            if lower > upper:
                continue
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue

            old_diffs = row_diffs[r_idx]
            # remove old
            for d in old_diffs:
                cnt = diff_counts[d]
                if cnt == 2:
                    collision_count -= 1
                diff_counts[d] = cnt - 1
                if cnt - 1 == 0:
                    del diff_counts[d]

            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs
            for d in new_diffs:
                if d in diff_counts:
                    if diff_counts[d] == 1:
                        collision_count += 1
                    diff_counts[d] += 1
                else:
                    diff_counts[d] = 1

            new_max_val = row[-1] if r_idx == (max(range(n), key=lambda i: rows[i][-1])) else max(r[-1] for r in rows)
            new_cost = collision_count * 10000 + new_max_val

            if new_cost < current_cost:
                current_cost = new_cost
                current_max_val = new_max_val
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = new_max_val
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    accept = random.random() < 0.7
                elif delta < 800:
                    accept = random.random() < math.exp(-delta / T)
                else:
                    accept = False
                if not accept:
                    row[i_idx] = old_val
                    for d in new_diffs:
                        cnt = diff_counts[d]
                        if cnt == 2:
                            collision_count -= 1
                        diff_counts[d] = cnt - 1
                        if cnt - 1 == 0:
                            del diff_counts[d]
                    for d in old_diffs:
                        if d in diff_counts:
                            if diff_counts[d] == 1:
                                collision_count += 1
                            diff_counts[d] += 1
                        else:
                            diff_counts[d] = 1
                    row_diffs[r_idx] = old_diffs

            T *= cooling
            if T < 0.1:
                T = 0.1

        return best_local_rows, best_local_max, best_local_cost

    # --- CSP with value ordering ---
    def csp_construct(target_M, time_limit):
        used = set()
        rows = []
        start_time = time.time()

        def dfs(row_idx):
            if time.time() - start_time > time_limit:
                return False
            if row_idx == n_rows:
                return True
            if target_M < k:
                return False
            # generate candidate rows in order of smallest max
            candidates = []
            for _ in range(60):
                pts = random.sample(range(1, target_M + 1), k)
                pts.sort()
                row = [0] + pts
                diffs = get_diffs(row)
                if len(set(diffs)) != len(diffs):
                    continue
                if any(d in used for d in diffs):
                    continue
                candidates.append((row, diffs))
            candidates.sort(key=lambda x: x[0][-1])
            for row, diffs in candidates:
                rows.append(row)
                used.update(diffs)
                if dfs(row_idx + 1):
                    return True
                rows.pop()
                for d in diffs:
                    used.discard(d)
            return False

        if dfs(0):
            return rows
        return None

    # Main loop
    while time.time() < deadline - 5.0:
        # 1. Try CSP for tighter scope
        for target in range(best_max - 1, best_max - 22, -1):
            if time.time() > deadline - 10.0:
                break
            csp_time = min(3.0, deadline - time.time() - 10.0)
            if csp_time <= 0.5:
                break
            result = csp_construct(target, csp_time)
            if result is not None:
                new_max = max(r[-1] for r in result)
                if new_max < best_max:
                    best_rows = result
                    best_max = new_max
                    break

        # 2. Polish with SA (multiple perturbed rounds)
        remaining = deadline - time.time() - 3.0
        if remaining <= 0:
            break
        for sa_round in range(3):
            sa_time = min(7.0, remaining / (3 - sa_round) - 0.5)
            if sa_time <= 0.5:
                break
            init_rows = [r[:] for r in best_rows]
            n_perturb = random.randint(2, min(5, n_rows * (row_len - 1)))
            for _ in range(n_perturb):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                row = init_rows[r_idx]
                lower = row[i_idx - 1] + 1
                upper = row[i_idx + 1] - 1 if i_idx < row_len - 1 else best_max + 8
                if lower <= upper:
                    init_rows[r_idx][i_idx] = random.randint(lower, upper)
            local_rows, local_max, local_cost = sa_optimize(init_rows, sa_time, best_max)
            if local_cost < 10000 and local_max < best_max:
                best_rows = local_rows
                best_max = local_max

    return best_rows
