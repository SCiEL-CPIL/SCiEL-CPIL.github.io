import time
import random

_START = None
_DEADLINE = 112.0  # safety margin vs 150s kill


def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + _DEADLINE

    n = k + 1
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(r[-1] for r in rows)

    def row_diffs(row):
        s = set()
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                s.add(row[j] - ai)
        return s

    def full_validate(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                ai = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ai
                    if diff in seen:
                        return None
                    seen.add(diff)
        return get_scope(rows)

    def greedy_init(seed=0):
        r = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
                    for skip in range(1, 6):
                        x2 = x + skip
                        new2 = {x2 - a for a in row}
                        if not (new2 & used) and not (new2 & {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}):
                            x = x2
                            break
            rows.append(row)
        return rows

    best_rows = greedy_init(0)
    best_scope = get_scope(best_rows)
    rng = random.Random(42)

    # ---------- Row reconstruction ----------
    def rebuild_row(used_other, max_val, jitter=0.0, seed=None):
        local_rng = random.Random(seed) if seed is not None else rng
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < n:
            found = False
            for trial_x in range(x + 1, max_val + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if jitter > 0 and local_rng.random() < jitter and trial_x < 300:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def local_polish(rows, max_passes=6):
        rows = [list(r) for r in rows]
        improved = True
        pc = 0
        while improved and time.time() < deadline - 12 and pc < max_passes:
            improved = False
            pc += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 12:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs(rows[j])
                best_row = list(rows[i])
                best_max = max(rows[i])
                for attempt in range(40):
                    if attempt < 10:
                        jitter = 0.0
                        max_v = best_max - 1
                    elif attempt < 25:
                        jitter = 0.15
                        max_v = best_max - 1
                    else:
                        jitter = 0.3
                        max_v = best_max + 5
                    new_row = rebuild_row(used_other, max_v, jitter=jitter)
                    if new_row is not None and max(new_row) < best_max:
                        best_row = new_row
                        best_max = max(new_row)
                if max(best_row) < max(rows[i]):
                    rows[i] = best_row
                    improved = True
        return rows

    # ---------- Targeted descent: shrink individual entries ----------
    def targeted_descent(rows, budget=3.0):
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False
        while time.time() - start < budget:
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if old_val <= lower:
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(8):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    row[pos2] = nv
                    scope = full_validate(cur)
                    if scope is not None and scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                    row[pos2] = old2
                if not improved:
                    break
                continue
            candidates = set()
            for v in range(lower, min(lower + 20, old_val)):
                candidates.add(v)
            for _ in range(12):
                candidates.add(rng.randint(lower, old_val - 1))
            for new_val in sorted(candidates, key=lambda x: -x):
                if new_val >= old_val:
                    continue
                row[pos] = new_val
                scope = full_validate(cur)
                if scope is not None:
                    if scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                row[pos] = old_val
            if not improved:
                break
        if improved:
            return cur, cur_scope
        return None, None

    # ---------- MRV backtracking ----------
    def mrv_solve(M, budget=2.5, seed=None):
        local_rng = random.Random(seed) if seed is not None else rng
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

    # ---------- Kick ----------
    def kick(rows, num_kicks=1):
        rows = [list(r) for r in rows]
        indices = rng.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs(rows[j])
            success = False
            for attempt in range(12):
                jitter = 0.3 + 0.1 * attempt
                new_row = rebuild_row(used_other, best_scope + 20, jitter=jitter)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, best_scope + 30, jitter=0.0)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

    # ---------- Main loop ----------
    it = 0
    while time.time() < deadline - 15:
        it += 1
        if it == 1:
            cur = [r[:] for r in best_rows]
        elif it <= 3:
            cur = greedy_init(seed=it * 1000)
        else:
            cur = [r[:] for r in best_rows]
            cur = kick(cur, num_kicks=rng.choice([1, 1, 2, 3]))

        cs = full_validate(cur)
        if cs is None:
            continue

        if time.time() < deadline - 15:
            polished = local_polish(cur)
            ps = full_validate(polished)
            if ps is not None:
                cur = polished
                cs = ps

        # Targeted descent
        if time.time() < deadline - 15:
            td_result = targeted_descent(cur, budget=2.0)
            if td_result is not None:
                td_rows, td_scope = td_result
                if td_scope < cs:
                    cur = td_rows
                    cs = td_scope

        # MRV tight-solve
        if time.time() < deadline - 15:
            target = min(best_scope - 1, cs - 1)
            if target >= 1:
                sol = mrv_solve(target, budget=2.0, seed=it)
                if sol is not None:
                    ss = full_validate(sol)
                    if ss is not None and ss < best_scope:
                        best_scope = ss
                        best_rows = [list(r) for r in sol]
                        cs = ss
                        cur = [list(r) for r in sol]

        if cs < best_scope:
            best_scope = cs
            best_rows = [list(r) for r in cur]

    # final polish
    if time.time() < deadline - 2:
        polished = local_polish(best_rows, max_passes=5)
        ps = full_validate(polished)
        if ps is not None and ps < best_scope:
            best_scope = ps
            best_rows = polished

    return best_rows
