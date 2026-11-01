import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def count_conflicts(rows):
        diff_counts = {}
        for r in rows:
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def greedy_init(rand_prob=0.1, window=20):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                candidates = []
                x = last + 1
                limit = 300
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

    def local_search_targeted(rows, time_limit):
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)

        while time.time() < start + time_limit:
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            if max_c_idx == 0:
                break
            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]
            all_diffs = set()
            for r in best_rows:
                for i in range(row_len):
                    for j in range(i + 1, row_len):
                        all_diffs.add(r[j] - r[i])
            remove_set = set()
            row = best_rows[max_r_idx]
            for i in range(max_c_idx):
                remove_set.add(current_val - row[i])
            for j in range(max_c_idx + 1, row_len):
                remove_set.add(row[j] - current_val)
            fixed_diffs = all_diffs - remove_set
            found = False
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                for i in range(max_c_idx):
                    if (new_val - row[i]) in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                for j in range(max_c_idx + 1, row_len):
                    if (row[j] - new_val) in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                new_left_diffs = [new_val - row[i] for i in range(max_c_idx)]
                new_right_diffs = [row[j] - new_val for j in range(max_c_idx + 1, row_len)]
                all_new = new_left_diffs + new_right_diffs
                if len(set(all_new)) != len(all_new):
                    continue
                row[max_c_idx] = new_val
                found = True
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                break
            if not found:
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, row_len - 1)
                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx - 1]
                next_v = best_rows[r_idx][c_idx + 1] if c_idx < row_len - 1 else curr_val + 100
                all_diffs = set()
                for r in best_rows:
                    for i in range(row_len):
                        for j in range(i + 1, row_len):
                            all_diffs.add(r[j] - r[i])
                remove_set = set()
                row = best_rows[r_idx]
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                for j in range(c_idx + 1, row_len):
                    remove_set.add(row[j] - curr_val)
                fixed_diffs = all_diffs - remove_set
                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len - 1 else curr_val + 50
                if low > high:
                    continue
                for _ in range(50):
                    cand = random.randint(low, high)
                    if cand == curr_val:
                        continue
                    valid = True
                    for i in range(c_idx):
                        if (cand - row[i]) in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    for j in range(c_idx + 1, row_len):
                        if (row[j] - cand) in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    new_left_diffs = [cand - row[i] for i in range(c_idx)]
                    new_right_diffs = [row[j] - cand for j in range(c_idx + 1, row_len)]
                    all_new = new_left_diffs + new_right_diffs
                    if len(set(all_new)) != len(all_new):
                        continue
                    row[c_idx] = cand
                    break
        return best_rows, best_scope

    def local_search_sa(rows, time_limit):
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        current_conflicts = count_conflicts(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        best_conflicts = current_conflicts
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        start = time.time()
        while time.time() < start + time_limit:
            if T < T_min:
                break
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            row = current_rows[r_idx]
            old_val = row[pos]
            lower_bound = row[pos - 1] + 1
            if pos < k:
                upper_bound = row[pos + 1] - 1
            else:
                upper_bound = current_scope + 10
            if lower_bound > upper_bound:
                continue
            v = random.randint(lower_bound, upper_bound)
            if v == old_val:
                continue
            row[pos] = v
            new_conflicts = count_conflicts(current_rows)
            new_scope = get_scope(current_rows)
            delta_conflict = new_conflicts - current_conflicts
            delta_scope = new_scope - current_scope
            accept = False
            if new_conflicts < current_conflicts:
                accept = True
            elif new_conflicts == current_conflicts:
                if new_scope < current_scope:
                    accept = True
                else:
                    if random.random() < math.exp(-delta_scope / T):
                        accept = True
            else:
                if random.random() < math.exp(-delta_conflict / T):
                    accept = True
            if accept:
                current_conflicts = new_conflicts
                current_scope = new_scope
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and current_scope < best_scope):
                    best_conflicts = current_conflicts
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                row[pos] = old_val
            T *= cooling
        return best_rows, best_scope

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def high_precision_polish(rows, time_limit):
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        pass_count = 0
        max_passes = 20
        while improved and pass_count < max_passes and (time.time() - start) < time_limit:
            improved = False
            pass_count += 1
            for r_idx in range(n_rows):
                if (time.time() - start) > time_limit:
                    break
                row = rows[r_idx]
                other_diffs = set()
                for j in range(n_rows):
                    if j != r_idx:
                        other_diffs |= row_diffs_set(rows[j])
                for c_idx in range(row_len - 1, 0, -1):
                    if (time.time() - start) > time_limit:
                        break
                    curr_val = row[c_idx]
                    prev_val = row[c_idx - 1]
                    remove_set = set()
                    for i in range(c_idx):
                        remove_set.add(curr_val - row[i])
                    for j in range(c_idx + 1, row_len):
                        remove_set.add(row[j] - curr_val)
                    fixed_diffs = other_diffs - remove_set
                    same_row_fixed = set()
                    for i in range(row_len):
                        if i == c_idx:
                            continue
                        for j in range(i + 1, row_len):
                            if j == c_idx:
                                continue
                            same_row_fixed.add(row[j] - row[i])
                    all_fixed = fixed_diffs | same_row_fixed
                    found = False
                    for new_val in range(curr_val - 1, prev_val, -1):
                        valid = True
                        for i in range(c_idx):
                            if (new_val - row[i]) in all_fixed:
                                valid = False
                                break
                        if not valid:
                            continue
                        for j in range(c_idx + 1, row_len):
                            if (row[j] - new_val) in all_fixed:
                                valid = False
                                break
                        if not valid:
                            continue
                        new_diffs = set()
                        for i in range(c_idx):
                            new_diffs.add(new_val - row[i])
                        for j in range(c_idx + 1, row_len):
                            new_diffs.add(row[j] - new_val)
                        if len(new_diffs) != len(new_diffs):
                            continue
                        if new_diffs & all_fixed:
                            continue
                        row[c_idx] = new_val
                        found = True
                        improved = True
                        break
                    if found:
                        other_diffs = set()
                        for j in range(n_rows):
                            if j != r_idx:
                                other_diffs |= row_diffs_set(rows[j])
        return rows

    def rebuild_row(used_other, max_val=None, jitter=0.0):
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < row_len:
            found = False
            search_end = max_val if max_val else (best_scope + 30)
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if jitter > 0 and random.random() < jitter and trial_x < 300:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def local_polish(rows, max_passes=4):
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time.time() < deadline - 10 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 10:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                best_row = list(rows[i])
                best_max = max(rows[i])
                new_row = rebuild_row(used_other, max_val=best_max - 1, jitter=0.0)
                if new_row is not None and max(new_row) < best_max:
                    best_row = new_row
                    best_max = max(new_row)
                for attempt in range(10):
                    jitter = 0.1 + 0.05 * attempt
                    new_row = rebuild_row(used_other, max_val=best_max - 1, jitter=jitter)
                    if new_row is not None and max(new_row) < best_max:
                        best_row = new_row
                        best_max = max(new_row)
                if max(best_row) < max(rows[i]):
                    rows[i] = best_row
                    improved = True
        return rows

    def kick(rows, num_kicks=1):
        rows = [list(r) for r in rows]
        indices = random.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            success = False
            for attempt in range(15):
                jitter = 0.3 + 0.1 * attempt
                new_row = rebuild_row(used_other, max_val=best_scope + 20, jitter=jitter)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, max_val=best_scope + 30, jitter=0.0)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

    # NEW: Gap-swap local search – operates on the gap representation of each row
    # A row is [0, g1, g1+g2, ..., sum(gaps)]. Swapping gaps between two rows
    # at two positions preserves row lengths and the global multiset of gaps,
    # changing only the distribution of prefix sums (i.e. the scope).
    def gap_swap_search(rows, time_limit):
        rows = [list(r) for r in rows]
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        T = 5.0
        T_min = 0.01
        cooling = 0.998

        def get_gaps(row):
            return [row[i + 1] - row[i] for i in range(row_len - 1)]

        def rebuild_from_gaps(gaps):
            row = [0]
            s = 0
            for g in gaps:
                s += g
                row.append(s)
            return row

        while time.time() < start + time_limit:
            if T < T_min:
                break
            # Pick two distinct rows and two positions in each
            r1, r2 = random.sample(range(n_rows), 2)
            p1 = random.randint(0, row_len - 2)
            p2 = random.randint(0, row_len - 2)
            g1 = get_gaps(rows[r1])
            g2 = get_gaps(rows[r2])
            old_g1, old_g2 = g1[p1], g2[p2]
            if old_g1 == old_g2:
                T *= cooling
                continue
            # Swap
            g1[p1], g2[p2] = old_g2, old_g1
            new_row1 = rebuild_from_gaps(g1)
            new_row2 = rebuild_from_gaps(g2)
            # Check increasing
            if any(new_row1[i + 1] <= new_row1[i] for i in range(row_len - 1)):
                continue
            if any(new_row2[i + 1] <= new_row2[i] for i in range(row_len - 1)):
                continue
            # Check distinctness of all differences
            rows[r1] = new_row1
            rows[r2] = new_row2
            all_diffs = set()
            conflict = False
            for r in rows:
                for i in range(row_len):
                    for j in range(i + 1, row_len):
                        d = r[j] - r[i]
                        if d in all_diffs:
                            conflict = True
                            break
                        all_diffs.add(d)
                if conflict:
                    break
            if conflict:
                rows[r1] = rebuild_from_gaps([old_g1 if i == p1 else g1[i] for i in range(row_len - 1)])
                rows[r2] = rebuild_from_gaps([old_g2 if i == p2 else g2[i] for i in range(row_len - 1)])
                T *= cooling
                continue
            new_scope = get_scope(rows)
            delta = new_scope - best_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
            elif random.random() < math.exp(-delta / T):
                pass  # accept, keep current
            else:
                # revert
                rows[r1] = rebuild_from_gaps([old_g1 if i == p1 else g1[i] for i in range(row_len - 1)])
                rows[r2] = rebuild_from_gaps([old_g2 if i == p2 else g2[i] for i in range(row_len - 1)])
            T *= cooling
        return best_rows, best_scope

    # NEW: MRV backtracking solver for a given scope bound M
    def mrv_solve(M, budget=3.0, seed=None):
        local_rng = random.Random(seed) if seed is not None else random
        start = time.time()
        used = set()
        rows = [[0] for _ in range(n_rows)]

        def candidates(row, M):
            last = row[-1]
            cands = []
            for v in range(last + 1, M + 1):
                new = {v - a for a in row}
                if (new & used) or (len(new) != len(row)):
                    continue
                cands.append(v)
            return cands

        def backtrack(rows, used, depth):
            if time.time() - start > budget:
                return None
            if depth == n_rows:
                if all(len(r) == n for r in rows):
                    return [list(r) for r in rows]
                return None
            best_i = -1
            best_cands = None
            best_len = 10 ** 9
            for i in range(n_rows):
                if len(rows[i]) >= n:
                    continue
                c = candidates(rows[i], M)
                if len(c) < best_len:
                    best_len = len(c)
                    best_i = i
                    best_cands = c
                    if best_len == 0:
                        break
            if best_i == -1:
                if all(len(r) == n for r in rows):
                    return [list(r) for r in rows]
                return None
            if best_len == 0:
                return None
            cands = list(best_cands)
            if local_rng.random() < 0.3:
                local_rng.shuffle(cands)
            else:
                cands.sort()
            row = rows[best_i]
            for v in cands:
                new = {v - a for a in row}
                row.append(v)
                used |= new
                res = backtrack(rows, used, depth + 1)
                if res is not None:
                    return res
                used.difference_update(new)
                row.pop()
            return None

        return backtrack(rows, used, 0)

    # --- MAIN SEARCH LOOP ---
    best_rows = None
    best_scope = float('inf')
    best_conflicts = float('inf')

    num_restarts = 6
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        rand_prob = 0.05 + 0.05 * i
        window = 10 + 5 * i
        rows = greedy_init(rand_prob, window)
        scope = get_scope(rows)
        conflicts = count_conflicts(rows)

        t1_rows, t1_scope = local_search_targeted(rows, 3.5)
        t1_conflicts = count_conflicts(t1_rows)
        if t1_conflicts < best_conflicts or (t1_conflicts == best_conflicts and t1_scope < best_scope):
            best_conflicts = t1_conflicts
            best_scope = t1_scope
            best_rows = t1_rows

        if time.time() > deadline - 20:
            break
        t2_rows, t2_scope = local_search_sa(t1_rows, 3.5)
        t2_conflicts = count_conflicts(t2_rows)
        if t2_conflicts < best_conflicts or (t2_conflicts == best_conflicts and t2_scope < best_scope):
            best_conflicts = t2_conflicts
            best_scope = t2_scope
            best_rows = t2_rows

        if best_conflicts == 0 and best_scope < 120:
            break

    if best_rows is not None:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 15:
            # ILS phase
            num_il_iters = int(remaining_time / 3.0)
            num_il_iters = max(1, num_il_iters)
            for it in range(num_il_iters):
                if time.time() > deadline - 15:
                    break
                cur = kick(best_rows, num_kicks=random.choice([1, 2]))
                cur_scope = get_scope(cur)
                cur_conflicts = count_conflicts(cur)
                polished = local_polish(cur, max_passes=3)
                pol_scope = get_scope(polished)
                pol_conflicts = count_conflicts(polished)
                if pol_conflicts < best_conflicts or (pol_conflicts == best_conflicts and pol_scope < best_scope):
                    best_conflicts = pol_conflicts
                    best_scope = pol_scope
                    best_rows = polished

    if best_rows is not None:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 10:
            # Gap-swap phase: new structural move
            half_time = remaining_time / 2
            gs_rows, gs_scope = gap_swap_search(best_rows, half_time)
            gs_conflicts = count_conflicts(gs_rows)
            if gs_conflicts < best_conflicts or (gs_conflicts == best_conflicts and gs_scope < best_scope):
                best_conflicts = gs_conflicts
                best_scope = gs_scope
                best_rows = gs_rows
            if time.time() < deadline - 5.0:
                f2_rows, f2_scope = local_search_targeted(best_rows, deadline - time.time() - 5.0)
                f2_conflicts = count_conflicts(f2_rows)
                if f2_conflicts < best_conflicts or (f2_conflicts == best_conflicts and f2_scope < best_scope):
                    best_conflicts = f2_conflicts
                    best_scope = f2_scope
                    best_rows = f2_rows

    if best_rows is None:
        best_rows = greedy_init(0.1, 20)

    # Final high-precision polish
    if time.time() < deadline - 3.0:
        polish_time = min(deadline - time.time() - 3.0, 25.0)
        polished = high_precision_polish(best_rows, polish_time)
        pol_conflicts = count_conflicts(polished)
        if pol_conflicts == 0:
            pol_scope = get_scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished

    # Try MRV solve with a tighter bound if we have time
    if time.time() < deadline - 5.0 and best_scope < 200:
        m = min(best_scope - 1, 140)
        result = mrv_solve(m, budget=min(deadline - time.time() - 5.0, 4.0))
        if result is not None:
            result_scope = get_scope(result)
            if result_scope < best_scope:
                best_scope = result_scope
                best_rows = result

    return best_rows
