import time, random, math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 114.0

    def remaining():
        return DEADLINE - (time.time() - t0)

    needed = k * (k + 1) // 2  # pairwise differences per row of length k+1

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    def sum_of(rows):
        return sum(sum(r) for r in rows)

    def obj(rows):
        return scope_of(rows) * 1000000 + sum_of(rows)

    def is_valid(rows):
        used = set()
        for r in rows:
            if r[0] != 0 or any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            d = row_diffs(r)
            if len(d) != needed:
                return False
            if d & used:
                return False
            used |= d
        return True

    # ---- deterministic greedy (baseline, always valid) ----
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # ---- randomized greedy: diversify the myopic "smallest" choice ----
    def rand_greedy(rng, bias, window):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                cands = []
                y = x + 1
                scanned = 0
                while len(cands) < window and scanned < 2000:
                    scanned += 1
                    new = {y - a for a in row}
                    if not (new & used):
                        cands.append(y)
                    y += 1
                while not cands:
                    new = {y - a for a in row}
                    if not (new & used):
                        cands.append(y)
                    y += 1
                if rng.random() < bias:
                    chosen = cands[0]
                else:
                    idx = int(len(cands) * (rng.random() ** 1.6))
                    idx = min(idx, len(cands) - 1)
                    chosen = cands[idx]
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen
            rows.append(row)
        return rows

    # ---- multi-objective local search: alternates scope-first and sum-first
    #      objectives, and adds a row-pair swap move absent from the parent. ----
    def local_search(start, rng, steps, T0=3.0):
        cur = [list(r) for r in start]
        rowd = [row_diffs(r) for r in cur]
        cur_used = set()
        for d in rowd:
            cur_used |= d

        row_sums = [sum(r) for r in cur]
        row_maxes = [max(r) for r in cur]
        cur_sum = sum(row_sums)
        cur_scope = max(row_maxes)

        # Pareto-style archive: keep best-by-scope and best-by-sum separately
        best_scope_rows = [list(r) for r in cur]
        best_scope_val = cur_scope
        best_sum_rows = [list(r) for r in cur]
        best_sum_val = cur_sum

        T = T0
        mode = 0  # 0 = scope-first, 1 = sum-first
        mode_len = 400

        for it in range(steps):
            if it % 200 == 0 and remaining() < 12:
                break
            # periodically flip objective to explore the Pareto front
            if it % mode_len == 0 and it > 0:
                mode = 1 - mode
                T = T0 * (0.6 + 0.4 * rng.random())

            # Objective function for acceptance
            def obj(rows, sums, maxes, scope, s):
                if mode == 0:
                    return scope * 1000000 + s
                else:
                    return s * 100 + scope

            cur_obj = obj(cur, row_sums, row_maxes, cur_scope, cur_sum)

            r = rng.randrange(n_rows)
            row = cur[r]

            # Move generation: 70% single-point, 30% row-pair swap
            if rng.random() < 0.70:
                # single-point move (as before)
                move_type = rng.random()
                if move_type < 0.5:
                    p = row.index(max(row))
                    if p == 0:
                        continue
                    lo = row[p - 1] + 1
                    hi = row[p] - 1
                    if lo > hi:
                        continue
                    v = rng.randrange(lo, hi + 1)
                    if v == row[p]:
                        continue
                elif move_type < 0.75:
                    p = row.index(max(row))
                    if p == 0:
                        continue
                    v = row[p] - 1
                    if v <= row[p - 1]:
                        continue
                else:
                    p = rng.randrange(1, len(row))
                    lo = row[p - 1] + 1
                    hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 30)
                    if lo > hi:
                        continue
                    v = rng.randrange(lo, hi + 1)
                    if v == row[p]:
                        continue

                new_row = row[:]
                new_row[p] = v
                if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                other = cur_used - rowd[r]
                if nd & other:
                    continue

                new_row_sums = row_sums[:]
                new_row_maxes = row_maxes[:]
                new_row_sums[r] = sum(new_row)
                new_row_maxes[r] = max(new_row)
                new_sum = cur_sum - row_sums[r] + new_row_sums[r]
                new_scope = max(new_row_maxes)
                new_obj = obj(None, new_row_sums, new_row_maxes, new_scope, new_sum)
                d = new_obj - cur_obj

                if d <= 0 or rng.random() < math.exp(-d / T):
                    cur[r] = new_row
                    rowd[r] = nd
                    cur_used = other | nd
                    row_sums[r] = new_row_sums[r]
                    row_maxes[r] = new_row_maxes[r]
                    cur_sum = new_sum
                    cur_scope = new_scope
                    # update Pareto archive
                    if cur_scope < best_scope_val:
                        best_scope_val = cur_scope
                        best_scope_rows = [list(x) for x in cur]
                    if cur_sum < best_sum_val:
                        best_sum_val = cur_sum
                        best_sum_rows = [list(x) for x in cur]
            else:
                # Row-pair swap: exchange the last two entries of two rows
                # to explore a different region of the solution space.
                if n_rows < 2:
                    continue
                r2 = rng.randrange(n_rows)
                if r2 == r:
                    continue
                row2 = cur[r2]
                # swap last two entries of row r and row r2
                if len(row) < 3 or len(row2) < 3:
                    continue
                new_row = row[:]
                new_row2 = row2[:]
                # swap the second-to-last and last positions
                new_row[-1], new_row2[-1] = new_row2[-1], new_row[-1]
                # check strict increase
                if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                if any(new_row2[i] >= new_row2[i + 1] for i in range(len(new_row2) - 1)):
                    continue
                nd1 = row_diffs(new_row)
                nd2 = row_diffs(new_row2)
                if len(nd1) != needed or len(nd2) != needed:
                    continue
                other = cur_used - rowd[r] - rowd[r2]
                if (nd1 | nd2) & other:
                    continue
                if nd1 & nd2:
                    continue

                new_row_sums = row_sums[:]
                new_row_maxes = row_maxes[:]
                new_row_sums[r] = sum(new_row)
                new_row_sums[r2] = sum(new_row2)
                new_row_maxes[r] = max(new_row)
                new_row_maxes[r2] = max(new_row2)
                new_sum = sum(new_row_sums)
                new_scope = max(new_row_maxes)
                new_obj = obj(None, new_row_sums, new_row_maxes, new_scope, new_sum)
                d = new_obj - cur_obj

                if d <= 0 or rng.random() < math.exp(-d / T):
                    cur[r] = new_row
                    cur[r2] = new_row2
                    rowd[r] = nd1
                    rowd[r2] = nd2
                    cur_used = other | nd1 | nd2
                    row_sums = new_row_sums
                    row_maxes = new_row_maxes
                    cur_sum = new_sum
                    cur_scope = new_scope
                    if cur_scope < best_scope_val:
                        best_scope_val = cur_scope
                        best_scope_rows = [list(x) for x in cur]
                    if cur_sum < best_sum_val:
                        best_sum_val = cur_sum
                        best_sum_rows = [list(x) for x in cur]

            T = max(0.05, T * 0.9995)

        # Return the better of the two Pareto-optimal candidates by scope
        if best_scope_val < best_sum_val:
            return best_scope_rows
        else:
            return best_sum_rows

    # ---- main search with explicit deadline guard ----
    base = greedy()
    best_rows = base
    best_scope = scope_of(base)
    rng = random.Random(20240607)

    # Phase 1: many fast randomized-greedy restarts
    while remaining() > 30:
        bias = rng.uniform(0.55, 0.97)
        window = rng.choice([3, 4, 5, 6, 8, 10])
        rows = rand_greedy(rng, bias, window)
        s = scope_of(rows)
        if s < best_scope:
            best_scope = s
            best_rows = rows
        # quick polish of promising candidates
        if s < best_scope + 6 and remaining() > 18:
            pol = local_search(rows, rng, 400)
            sp = scope_of(pol)
            if sp < best_scope:
                best_scope = sp
                best_rows = pol

    # Phase 2: deeper annealing from the best + perturbed restarts
    for _ in range(6):
        if remaining() < 8:
            break
        res = local_search(best_rows, rng, 3500, T0=rng.uniform(2.0, 6.0))
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res
        else:
            # perturb the best to escape its basin, then re-anneal
            pert = [list(r) for r in best_rows]
            for _ in range(4):
                rr = rng.randrange(n_rows)
                pp = rng.randrange(1, len(pert[rr]))
                lo = pert[rr][pp - 1] + 1
                hi = (pert[rr][pp + 1] - 1) if pp + 1 < len(pert[rr]) else best_scope + 25
                if lo > hi:
                    continue
                nv = rng.randrange(lo, hi + 1)
                new_row = pert[rr][:]
                new_row[pp] = nv
                if not all(new_row[i] < new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                used, ok = set(), True
                for i, row in enumerate(pert):
                    d = nd if i == rr else row_diffs(row)
                    if len(d) != needed or (d & used):
                        ok = False
                        break
                    used |= d
                if ok:
                    pert[rr] = new_row
                    break
            res = local_search(pert, rng, 2500, T0=4.0)
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    # Final safety: guarantee a valid result
    if not is_valid(best_rows):
        best_rows = base
    return best_rows
