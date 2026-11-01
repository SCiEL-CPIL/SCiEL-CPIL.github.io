import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    row_len = k + 1
    total_diffs = n_rows * k * (k + 1) // 2

    def get_scope(rows):
        return max(max(r) for r in rows)

    def count_conflicts(rows):
        diff_counts = {}
        for r in rows:
            for i in range(row_len):
                for j in range(i+1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def all_diffs_set(rows):
        s = set()
        for r in rows:
            for i in range(row_len):
                for j in range(i+1, row_len):
                    s.add(r[j] - r[i])
        return s

    def greedy_init(rand_prob=0.05, window=15):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                candidates = []
                x = last + 1
                limit = 400
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
            rows.append(row)
        return rows

    def targeted_descent(rows, time_limit):
        """Try to reduce each element from right to left in each row."""
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        pass_count = 0
        max_passes = 30
        while improved and pass_count < max_passes and (time.time() - start) < time_limit:
            improved = False
            pass_count += 1
            for r_idx in range(n_rows):
                if (time.time() - start) > time_limit:
                    break
                row = rows[r_idx]
                # Collect all diffs from other rows
                other_diffs = set()
                for j in range(n_rows):
                    if j != r_idx:
                        for i in range(row_len):
                            for jj in range(i+1, row_len):
                                other_diffs.add(rows[j][jj] - rows[j][i])
                # Same-row diffs excluding current element
                for c_idx in range(row_len - 1, 0, -1):
                    if (time.time() - start) > time_limit:
                        break
                    curr_val = row[c_idx]
                    prev_val = row[c_idx - 1]
                    # Build set of fixed diffs (other rows + same row excluding c_idx)
                    fixed = set(other_diffs)
                    for i in range(row_len):
                        if i == c_idx:
                            continue
                        for j in range(i+1, row_len):
                            if j == c_idx:
                                continue
                            fixed.add(row[j] - row[i])
                    # Try to find smaller value
                    found = False
                    for new_val in range(curr_val - 1, prev_val, -1):
                        ok = True
                        new_diffs = []
                        for i in range(c_idx):
                            d = new_val - row[i]
                            if d in fixed:
                                ok = False
                                break
                            new_diffs.append(d)
                        if not ok:
                            continue
                        for j in range(c_idx + 1, row_len):
                            d = row[j] - new_val
                            if d in fixed:
                                ok = False
                                break
                            new_diffs.append(d)
                        if not ok:
                            continue
                        if len(set(new_diffs)) != len(new_diffs):
                            continue
                        row[c_idx] = new_val
                        found = True
                        improved = True
                        break
        return rows

    def simulated_annealing(rows, time_limit):
        """SA with element moves."""
        rows = [list(r) for r in rows]
        start = time.time()
        current_conflicts = count_conflicts(rows)
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_conflicts = current_conflicts
        best_scope = current_scope
        T = 20.0
        T_min = 0.01
        cooling = 0.998
        no_improve = 0
        max_no_improve = 5000

        while (time.time() - start) < time_limit and T > T_min and no_improve < max_no_improve:
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, row_len - 1)
            row = rows[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            upper = row[pos + 1] - 1 if pos < row_len - 1 else current_scope + 20
            if lower > upper:
                continue
            v = random.randint(lower, upper)
            if v == old_val:
                continue
            row[pos] = v
            new_conflicts = count_conflicts(rows)
            new_scope = get_scope(rows)
            delta_c = new_conflicts - current_conflicts
            delta_s = new_scope - current_scope
            accept = False
            if new_conflicts < current_conflicts:
                accept = True
            elif new_conflicts == current_conflicts:
                if new_scope < current_scope:
                    accept = True
                elif random.random() < math.exp(-delta_s / max(T, 0.01)):
                    accept = True
            else:
                if random.random() < math.exp(-(delta_c * 2 + delta_s) / max(T, 0.01)):
                    accept = True
            if accept:
                current_conflicts = new_conflicts
                current_scope = new_scope
                if (new_conflicts < best_conflicts or
                    (new_conflicts == best_conflicts and new_scope < best_scope)):
                    best_conflicts = new_conflicts
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                row[pos] = old_val
                no_improve += 1
            T *= cooling
        return best_rows

    def rebuild_row(used_other, max_val=None, jitter=0.0):
        """Rebuild a single row greedily, avoiding used_other diffs."""
        row = [0]
        row_diffs = set()
        while len(row) < row_len:
            last = row[-1]
            found = False
            search_end = max_val if max_val else 500
            for trial in range(last + 1, search_end + 1):
                new_diffs = {trial - a for a in row}
                if not (new_diffs & used_other) and not (new_diffs & row_diffs):
                    if jitter > 0 and random.random() < jitter:
                        continue
                    row.append(trial)
                    row_diffs |= new_diffs
                    found = True
                    break
            if not found:
                # Try harder
                for trial in range(search_end + 1, search_end + 200):
                    new_diffs = {trial - a for a in row}
                    if not (new_diffs & used_other) and not (new_diffs & row_diffs):
                        row.append(trial)
                        row_diffs |= new_diffs
                        found = True
                        break
            if not found:
                return None
        return row

    def row_polish(rows, time_limit):
        """Try to rebuild each row to reduce its max element."""
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        pass_count = 0
        max_passes = 5
        while improved and pass_count < max_passes and (time.time() - start) < time_limit:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if (time.time() - start) > time_limit:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        for a in range(row_len):
                            for b in range(a+1, row_len):
                                used_other.add(rows[j][b] - rows[j][a])
                best_max = max(rows[i])
                # Try greedy rebuild
                new_row = rebuild_row(used_other, max_val=best_max - 1, jitter=0.0)
                if new_row is not None and max(new_row) < best_max:
                    rows[i] = new_row
                    best_max = max(new_row)
                    improved = True
                # Try with jitter
                for att in range(5):
                    if (time.time() - start) > time_limit:
                        break
                    jitter = 0.15 + 0.1 * att
                    new_row = rebuild_row(used_other, max_val=best_max - 1, jitter=jitter)
                    if new_row is not None and max(new_row) < best_max:
                        rows[i] = new_row
                        best_max = max(new_row)
                        improved = True
        return rows

    def kick_and_polish(rows, time_limit):
        """Perturb 1-2 rows then polish."""
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    for a in range(row_len):
                        for b in range(a+1, row_len):
                            used_other.add(rows[j][b] - rows[j][a])
            success = False
            for att in range(10):
                jitter = 0.2 + 0.1 * att
                new_row = rebuild_row(used_other, max_val=500, jitter=jitter)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, max_val=500, jitter=0.0)
                if new_row is not None:
                    rows[idx] = new_row
        # Polish
        remaining = time_limit - (time.time() - (time.time() - time_limit))
        # Just do a quick targeted descent
        rows = targeted_descent(rows, min(3.0, time_limit * 0.4))
        rows = row_polish(rows, min(3.0, time_limit * 0.4))
        return rows

    # --- MAIN SEARCH ---
    best_rows = None
    best_scope = float('inf')
    best_conflicts = float('inf')

    # Phase 1: Multiple restarts with targeted descent + SA
    num_restarts = 6
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        rand_prob = 0.03 + 0.04 * i
        window = 10 + 4 * i
        rows = greedy_init(rand_prob, window)
        scope = get_scope(rows)
        conflicts = count_conflicts(rows)

        # Targeted descent
        t1 = time.time()
        t1_rows = targeted_descent(rows, 5.0)
        t1_conflicts = count_conflicts(t1_rows)
        t1_scope = get_scope(t1_rows)
        if t1_conflicts < best_conflicts or (t1_conflicts == best_conflicts and t1_scope < best_scope):
            best_conflicts = t1_conflicts
            best_scope = t1_scope
            best_rows = t1_rows

        if time.time() > deadline - 20:
            break

        # SA
        t2_rows = simulated_annealing(t1_rows, 5.0)
        t2_conflicts = count_conflicts(t2_rows)
        t2_scope = get_scope(t2_rows)
        if t2_conflicts < best_conflicts or (t2_conflicts == best_conflicts and t2_scope < best_scope):
            best_conflicts = t2_conflicts
            best_scope = t2_scope
            best_rows = t2_rows

        if best_conflicts == 0 and best_scope < 118:
            break

    if best_rows is None:
        best_rows = greedy_init(0.1, 15)
        best_rows = targeted_descent(best_rows, 10.0)
        best_scope = get_scope(best_rows)
        best_conflicts = count_conflicts(best_rows)

    # Phase 2: ILS with kick + polish
    remaining = deadline - time.time() - 10
    if remaining > 15:
        num_iters = int(remaining / 8.0)
        num_iters = max(1, num_iters)
        for it in range(num_iters):
            if time.time() > deadline - 10:
                break
            it_time = min(8.0, deadline - time.time() - 10)
            if it_time < 3:
                break
            cur = kick_and_polish(best_rows, it_time)
            cur_scope = get_scope(cur)
            cur_conflicts = count_conflicts(cur)
            if cur_conflicts < best_conflicts or (cur_conflicts == best_conflicts and cur_scope < best_scope):
                best_conflicts = cur_conflicts
                best_scope = cur_scope
                best_rows = cur

    # Phase 3: Final polish
    remaining = deadline - time.time() - 3
    if remaining > 5:
        f1 = targeted_descent(best_rows, min(remaining * 0.5, 15.0))
        f1_conflicts = count_conflicts(f1)
        f1_scope = get_scope(f1)
        if f1_conflicts < best_conflicts or (f1_conflicts == best_conflicts and f1_scope < best_scope):
            best_conflicts = f1_conflicts
            best_scope = f1_scope
            best_rows = f1

        remaining2 = deadline - time.time() - 2
        if remaining2 > 3:
            f2 = row_polish(best_rows, min(remaining2 * 0.5, 10.0))
            f2_conflicts = count_conflicts(f2)
            f2_scope = get_scope(f2)
            if f2_conflicts < best_conflicts or (f2_conflicts == best_conflicts and f2_scope < best_scope):
                best_conflicts = f2_conflicts
                best_scope = f2_scope
                best_rows = f2

    # Final targeted descent sweep
    remaining = deadline - time.time() - 2
    if remaining > 2:
        best_rows = targeted_descent(best_rows, remaining)

    return best_rows
