import time
import random
import math
import sys

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 140.0
    m = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def time_ok(margin=5.0):
        return time.time() < deadline - margin

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def diffs_to_bitmask(diffs, max_d):
        bm = 0
        for d in diffs:
            if d <= max_d:
                bm |= (1 << d)
        return bm

    def bitmask_has_conflict(bm, diffs):
        for d in diffs:
            if bm & (1 << d):
                return True
        return False

    # ---- Core: generate a row given used diffs bitmask and max value ----
    def gen_row(used_bm, max_val, k_val, rng, p_small=0.7, window=3):
        """Generate a row of k_val+1 entries starting at 0, all <= max_val.
        used_bm: bitmask of forbidden differences.
        Returns row list or None."""
        row = [0]
        cur_bm = 0  # diffs within current row
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            while x <= max_val:
                # new diffs: x - a for a in row
                new_diffs = [x - a for a in row]
                # check conflicts
                conflict = False
                for d in new_diffs:
                    if (used_bm | cur_bm) & (1 << d):
                        conflict = True
                        break
                if not conflict:
                    # check new diffs are all distinct
                    if len(set(new_diffs)) == len(new_diffs):
                        candidates.append(x)
                        if len(candidates) >= window:
                            break
                x += 1
            if not candidates:
                return None
            if rng.random() < p_small:
                nxt = candidates[0]
            else:
                nxt = rng.choice(candidates)
            row.append(nxt)
            for a in row[:-1]:
                cur_bm |= (1 << (nxt - a))
        return row

    # ---- Tighten: pull elements down ----
    def tighten(rows, max_iter=300):
        rows = [list(r) for r in rows]
        for _ in range(max_iter):
            if not time_ok(15.0):
                break
            improved = False
            for r in range(n_rows):
                row = rows[r]
                for pos in range(len(row) - 1, 0, -1):
                    target = row[pos - 1] + 1
                    if target >= row[pos]:
                        continue
                    cand = row[:pos] + [target] + row[pos + 1:]
                    ok = True
                    for i in range(1, len(cand)):
                        if cand[i] <= cand[i - 1]:
                            ok = False
                            break
                    if not ok:
                        continue
                    idiffs = set()
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            d = cand[j] - cand[i]
                            if d in idiffs:
                                ok = False
                                break
                            idiffs.add(d)
                        if not ok:
                            break
                    if not ok:
                        continue
                    for idx in range(n_rows):
                        if idx == r:
                            continue
                        od = row_diffs(rows[idx])
                        if idiffs & od:
                            ok = False
                            break
                    if ok:
                        row[pos] = target
                        improved = True
            if not improved:
                break
        return rows

    # ---- Greedy initial with symmetry breaking ----
    def greedy_initial(rng, max_val=400):
        used_bm = 0
        rows = []
        prev_first = 0
        for r in range(n_rows):
            row = None
            for _ in range(100):
                row = gen_row(used_bm, max_val, k, rng, p_small=0.85, window=3)
                if row is not None:
                    if r > 0 and row[1] <= prev_first:
                        row = None
                        continue
                    break
            if row is None:
                row = gen_row(used_bm, max_val + 200, k, rng, p_small=0.9, window=2)
            if row is None:
                return None
            rows.append(row)
            prev_first = row[1]
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_bm |= (1 << (row[j] - row[i]))
        return rows

    # ---- Backtracking with constraint propagation ----
    def backtrack_scope(target_scope, rng, max_nodes=80000):
        """Try to construct all rows with max entry <= target_scope."""
        used_bm = 0
        rows = []
        nodes = [0]
        max_d = target_scope

        def build_row(row_idx, gap_pos, cur_gaps, cur_span, cur_bm_row):
            if not time_ok(10.0):
                return False
            nodes[0] += 1
            if nodes[0] > max_nodes:
                return False

            if gap_pos == k:
                row = [0]
                s = 0
                for g in cur_gaps:
                    s += g
                    row.append(s)
                rows.append(row)
                return True

            remaining = k - gap_pos
            min_add = remaining
            max_add = target_scope - cur_span
            if max_add < min_add:
                return False

            # Try gap values
            max_gap = max_add if gap_pos == k - 1 else max_add - (remaining - 1)
            if max_gap < 1:
                return False

            # Collect valid candidates
            valid = []
            for g in range(1, max_gap + 1):
                # New diffs: g + suffix sums of cur_gaps
                new_diffs = []
                suffix = 0
                for i in range(gap_pos - 1, -1, -1):
                    suffix += cur_gaps[i]
                    d = g + suffix
                    new_diffs.append(d)
                # The diff to position 0 is cur_span + g = suffix + g (when suffix = sum all)
                # Actually new_diffs already includes g + sum(cur_gaps) = g + cur_span
                # Wait: suffix after full loop = sum(cur_gaps) = cur_span
                # So new_diffs = [g+cur_gaps[gap_pos-1], g+cur_gaps[gap_pos-1]+cur_gaps[gap_pos-2], ..., g+cur_span]
                # That's correct: these are the diffs from new position to each previous position

                # Check conflicts
                conflict = False
                for d in new_diffs:
                    if d > max_d:
                        conflict = True
                        break
                    if (used_bm | cur_bm_row) & (1 << d):
                        conflict = True
                        break
                if conflict:
                    continue
                # Check new_diffs are distinct
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                valid.append(g)

            if not valid:
                return False

            # Order: prefer smaller gaps with some randomness
            if rng.random() < 0.7:
                valid.sort()
            else:
                rng.shuffle(valid)

            for g in valid:
                new_span = cur_span + g
                # Update row bitmask
                new_bm = cur_bm_row
                suffix = 0
                for i in range(gap_pos - 1, -1, -1):
                    suffix += cur_gaps[i]
                    d = g + suffix
                    new_bm |= (1 << d)

                if build_row(row_idx, gap_pos + 1, cur_gaps + [g], new_span, new_bm):
                    return True

            return False

        for r in range(n_rows):
            if not build_row(r, 0, [], 0, 0):
                return None

        return rows

    # ---- Local refinement: move individual elements ----
    def local_refine(rows, rng, max_iter=8000):
        rows = [list(r) for r in rows]
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        for _ in range(max_iter):
            if not time_ok(10.0):
                break
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            for delta in [-1, 1, -2, 2]:
                new_val = row[pos] + delta
                if new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                test_row = row[:]
                test_row[pos] = new_val
                idiffs = set()
                ok = True
                for i in range(len(test_row)):
                    for j in range(i + 1, len(test_row)):
                        d = test_row[j] - test_row[i]
                        if d in idiffs:
                            ok = False
                            break
                        idiffs.add(d)
                    if not ok:
                        break
                if not ok:
                    continue
                conflict = False
                for idx in range(n_rows):
                    if idx == r_idx:
                        continue
                    if idiffs & row_diffs(rows[idx]):
                        conflict = True
                        break
                if conflict:
                    continue
                other_max = max(rows[i][-1] for i in range(n_rows) if i != r_idx)
                new_scope = max(other_max, test_row[-1])
                if new_scope <= current_max:
                    rows[r_idx] = test_row
                    current_max = new_scope
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in rows]
                    break
        return best_rows

    # ---- Simulated annealing with row regeneration ----
    def anneal(rows, rng, max_iter=12000):
        rows = [list(r) for r in rows]
        used_bm = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used_bm |= (1 << (r[j] - r[i]))
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        T = 6.0
        cooling = 0.9985
        min_T = 0.2

        for _ in range(max_iter):
            if not time_ok(12.0):
                break
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            # Remove old row's diffs from used_bm
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    used_bm &= ~(1 << (old_row[j] - old_row[i]))

            p_sm = 0.6 + 0.35 * rng.random()
            win = 2 + rng.randint(0, 3)
            new_row = gen_row(used_bm, current_max + 20, k, rng, p_small=p_sm, window=win)

            if new_row is None:
                # Restore
                for i in range(len(old_row)):
                    for j in range(i + 1, len(old_row)):
                        used_bm |= (1 << (old_row[j] - old_row[i]))
                continue

            new_diffs_set = row_diffs(new_row)
            other_max = max(rows[i][-1] for i in range(n_rows) if i != r_idx)
            new_global = max(other_max, new_row[-1])
            delta = new_global - current_max

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True
            else:
                prob = math.exp(-delta / max(T, 0.01))
                accept = rng.random() < prob

            if accept:
                rows[r_idx] = new_row
                for d in new_diffs_set:
                    used_bm |= (1 << d)
                current_max = new_global
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            else:
                for i in range(len(old_row)):
                    for j in range(i + 1, len(old_row)):
                        used_bm |= (1 << (old_row[j] - old_row[i]))

            T = max(min_T, T * cooling)

        return best_rows

    # ---- Main search ----
    rng = random.Random(42)

    # 1. Greedy initial
    best_rows = greedy_initial(rng, max_val=350)
    if best_rows is None or not is_valid(best_rows):
        best_rows = greedy_initial(rng, max_val=500)
    if best_rows is None or not is_valid(best_rows):
        # Absolute fallback: very loose
        used_bm = 0
        best_rows = []
        for _ in range(n_rows):
            row = gen_row(used_bm, 800, k, rng, p_small=0.95, window=2)
            if row is None:
                row = [0] + list(range(1, k + 1)) * 10  # should not happen
            best_rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_bm |= (1 << (row[j] - row[i]))

    # 2. Tighten
    best_rows = tighten(best_rows)
    if not is_valid(best_rows):
        best_rows = greedy_initial(rng, max_val=400)
        best_rows = tighten(best_rows)
    best_scope = scope_of(best_rows)

    # 3. Anneal
    rows_a = anneal(best_rows, rng, max_iter=10000)
    if is_valid(rows_a):
        s = scope_of(rows_a)
        if s < best_scope:
            best_rows = rows_a
            best_scope = s

    # 4. Local refine
    rows_l = local_refine(best_rows, rng, max_iter=5000)
    if is_valid(rows_l):
        s = scope_of(rows_l)
        if s < best_scope:
            best_rows = rows_l
            best_scope = s

    # 5. Tighten again
    best_rows = tighten(best_rows)
    if is_valid(best_rows):
        best_scope = scope_of(best_rows)

    # 6. Backtracking for lower scope
    target = best_scope - 1
    max_bt = 25
    bt_count = 0
    while target > 100 and bt_count < max_bt and time_ok(30.0):
        bt_count += 1
        rng_bt = random.Random(5000 + bt_count * 31337)
        time_left = deadline - 30 - time.time()
        est_nodes = max(5000, min(60000, int(time_left * 400)))
        sol = backtrack_scope(target, rng_bt, max_nodes=est_nodes)
        if sol and is_valid(sol):
            new_scope = scope_of(sol)
            if new_scope < best_scope:
                best_rows = sol
                best_scope = new_scope
                best_rows = tighten(best_rows)
                if is_valid(best_rows):
                    best_scope = scope_of(best_rows)
                rows_ref = local_refine(best_rows, rng_bt, max_iter=2000)
                if is_valid(rows_ref):
                    s_ref = scope_of(rows_ref)
                    if s_ref < best_scope:
                        best_rows = rows_ref
                        best_scope = s_ref
                target = best_scope - 1
            else:
                target -= 1
        else:
            target -= 1

    # 7. Diversified restarts
    for trial in range(6):
        if not time_ok(25.0):
            break
        rng2 = random.Random(2000 + trial * 7919)
        rows = greedy_initial(rng2, max_val=300 + trial * 20)
        if rows is None or not is_valid(rows):
            continue
        rows = tighten(rows)
        if not is_valid(rows):
            continue
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s
        rows2 = anneal(rows, rng2, max_iter=6000)
        if is_valid(rows2):
            s2 = scope_of(rows2)
            if s2 < best_scope:
                best_rows = rows2
                best_scope = s2
        rows3 = local_refine(best_rows, rng2, max_iter=2500)
        if is_valid(rows3):
            s3 = scope_of(rows3)
            if s3 < best_scope:
                best_rows = rows3
                best_scope = s3

    # Final tighten
    best_rows = tighten(best_rows, max_iter=200)
    if not is_valid(best_rows):
        best_rows = greedy_initial(random.Random(7), max_val=500)
        best_rows = tighten(best_rows)

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
