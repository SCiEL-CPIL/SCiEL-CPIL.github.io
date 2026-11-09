import time, random, math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 118.0

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

    # ---- Row reconstruction with backtracking ----
    def reconstruct_row(blocked, rng, max_scope):
        """Build a row avoiding diffs in `blocked`, starting at 0, k+1 elements.
        Uses backtracking to find a valid row within the given scope."""
        row = [0]
        used_local = set()
        
        def backtrack(pos, current_row, current_used):
            if pos == k + 1:
                return current_row[:]
            start = current_row[-1] + 1
            limit = min(max_scope, 200)
            for y in range(start, limit + 1):
                new = {y - a for a in current_row}
                if not (new & blocked) and not (new & current_used):
                    result = backtrack(pos + 1, current_row + [y], current_used | new)
                    if result is not None:
                        return result
            return None
        
        result = backtrack(1, [0], set())
        return result

    # ---- Fast local search focused on scope reduction ----
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
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0

        for it in range(steps):
            if it % 200 == 0 and remaining() < 12:
                break
            r = rng.randrange(n_rows)
            row = cur[r]
            row_max = row_maxes[r]
            
            # Focus on reducing the maximum entry
            if row_max == cur_scope:
                # This row has the global max; try to reduce it
                p = len(row) - 1  # last element is the max
                lo = row[p - 1] + 1
                hi = row[p] - 1
                if lo <= hi:
                    v = rng.randrange(lo, hi + 1)
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
                else:
                    # Can't reduce max directly; try interior moves
                    p = rng.randrange(1, len(row))
                    lo = row[p - 1] + 1
                    hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 20)
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
            else:
                # Not the max row; try to improve it or help free up space
                move_type = rng.random()
                if move_type < 0.5:
                    # Random interior mutation
                    p = rng.randrange(1, len(row))
                    lo = row[p - 1] + 1
                    hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 20)
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
                else:
                    # Try to reduce this row's max
                    p = len(row) - 1
                    lo = row[p - 1] + 1
                    hi = row[p] - 1
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

            new_max_row = max(new_row)
            new_sum_row = sum(new_row)

            if new_max_row < row_maxes[r]:
                if row_maxes[r] == cur_scope:
                    others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                    new_scope = max(new_max_row, others_max)
                else:
                    new_scope = cur_scope
            else:
                new_scope = max(cur_scope, new_max_row)

            new_sum = cur_sum - row_sums[r] + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            d = new_obj - cur_obj

            if d <= 0 or (T > 0.01 and rng.random() < math.exp(-d / T)):
                cur[r] = new_row
                rowd[r] = nd
                cur_used = other | nd
                row_sums[r] = new_sum_row
                row_maxes[r] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]
            T = max(0.05, T * 0.9995)

        return best

    # ---- Targeted scope reduction: try to fit all rows within a given scope ----
    def try_scope_reduction(rows, target_scope, rng, iterations=5000):
        """Try to reduce the scope of rows to target_scope or below."""
        cur = [list(r) for r in rows]
        rowd = [row_diffs(r) for r in cur]
        cur_used = set()
        for d in rowd:
            cur_used |= d

        row_maxes = [max(r) for r in cur]
        cur_scope = max(row_maxes)
        
        if cur_scope <= target_scope:
            return cur

        best = [list(r) for r in cur]
        best_scope = cur_scope
        T = 2.0

        for it in range(iterations):
            if it % 300 == 0 and remaining() < 15:
                break
            
            # Find rows that exceed target_scope
            candidates = [i for i in range(n_rows) if row_maxes[i] > target_scope]
            if not candidates:
                break
            
            r = rng.choice(candidates)
            row = cur[r]
            row_max = row_maxes[r]
            
            # Try to reduce the max element
            p = len(row) - 1
            lo = row[p - 1] + 1
            hi = min(row[p] - 1, target_scope)
            if lo > hi:
                # Can't fit in target; try interior moves
                p = rng.randrange(1, len(row))
                lo = row[p - 1] + 1
                hi = (row[p + 1] - 1) if p + 1 < len(row) else target_scope
                if lo > hi:
                    continue
                v = rng.randrange(lo, hi + 1)
                if v == row[p]:
                    continue
                new_row = row[:]
                new_row[p] = v
            else:
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
            
            new_max_row = max(new_row)
            new_sum_row = sum(new_row)
            
            if new_max_row < row_maxes[r]:
                others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                new_scope = max(new_max_row, others_max)
            else:
                new_scope = max(cur_scope, new_max_row)
            
            new_sum = sum(sum(r) for r in cur) - sum(row) + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            cur_obj = cur_scope * 1000000 + sum(sum(r) for r in cur)
            d = new_obj - cur_obj

            if d <= 0 or (T > 0.01 and rng.random() < math.exp(-d / T)):
                cur[r] = new_row
                rowd[r] = nd
                cur_used = other | nd
                row_maxes[r] = new_max_row
                cur_scope = new_scope
                if cur_scope < best_scope:
                    best_scope = cur_scope
                    best = [list(x) for x in cur]
            T = max(0.05, T * 0.999)

        return best

    # ---- Main search ----
    base = greedy()
    best_rows = base
    best_scope = scope_of(base)
    rng = random.Random(20240607)

    # Phase 1: many fast randomized-greedy restarts
    while remaining() > 40:
        bias = rng.uniform(0.55, 0.97)
        window = rng.choice([3, 4, 5, 6, 8, 10])
        rows = rand_greedy(rng, bias, window)
        s = scope_of(rows)
        if s < best_scope:
            best_scope = s
            best_rows = rows
        # Quick polish of promising candidates
        if s < best_scope + 10 and remaining() > 25:
            pol = local_search(rows, rng, 800)
            sp = scope_of(pol)
            if sp < best_scope:
                best_scope = sp
                best_rows = pol

    # Phase 2: targeted scope reduction attempts
    # Try to reduce scope by 1, 2, 3... at a time
    target = best_scope - 1
    while target >= max(112, best_scope - 15) and remaining() > 15:
        res = try_scope_reduction(best_rows, target, rng, iterations=8000)
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res
            target = best_scope - 1
        else:
            target -= 1

    # Phase 3: deeper annealing from the best
    for _ in range(6):
        if remaining() < 12:
            break
        res = local_search(best_rows, rng, 5000, T0=rng.uniform(2.0, 5.0))
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res
        else:
            # Perturb: try reconstructing a random row, then re-anneal
            pert = [list(r) for r in best_rows]
            rr = rng.randrange(n_rows)
            other = set()
            for i, row in enumerate(pert):
                if i != rr:
                    other |= row_diffs(row)
            new_row = reconstruct_row(other, rng, best_scope + 5)
            if new_row is not None:
                nd = row_diffs(new_row)
                if len(nd) == needed and not (nd & other):
                    pert[rr] = new_row
            res = local_search(pert, rng, 4000, T0=4.0)
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    # Final safety: guarantee a valid result
    if not is_valid(best_rows):
        best_rows = base
    return best_rows
