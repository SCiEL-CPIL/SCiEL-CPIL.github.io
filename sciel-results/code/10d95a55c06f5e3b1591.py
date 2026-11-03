import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    L = k + 1
    num_diffs = n_rows * k * (k + 1) // 2
    lb = num_diffs  # counting lower bound (scope must be at least this)

    best_rows = None
    best_scope = float('inf')

    # ---------- helpers ----------
    def build_used(rows):
        used = set()
        for row in rows:
            for i in range(L):
                for j in range(i + 1, L):
                    used.add(row[j] - row[i])
        return used

    def diffs_at(row, pos):
        d = set()
        for i in range(L):
            if i < pos:
                d.add(row[pos] - row[i])
            elif i > pos:
                d.add(row[i] - row[pos])
        return d

    def can_set(row, pos, v, used):
        if pos > 0 and v <= row[pos - 1]:
            return False
        if pos < L - 1 and v >= row[pos + 1]:
            return False
        old = diffs_at(row, pos)
        nd = set()
        for i in range(L):
            if i < pos:
                nd.add(v - row[i])
            elif i > pos:
                nd.add(row[i] - v)
        return not (nd & (used - old))

    def apply_set(row, pos, v, used):
        old = diffs_at(row, pos)
        nd = set()
        for i in range(L):
            if i < pos:
                nd.add(v - row[i])
            elif i > pos:
                nd.add(row[i] - v)
        for d in old:
            used.discard(d)
        for d in nd:
            used.add(d)
        row[pos] = v

    def greedy_baseline(rng=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # ---------- local compaction (pull each element to smallest feasible) ----------
    def compact_pass(rows, seed=0, rounds=1000):
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        used = build_used(rows)
        improved = True
        r = 0
        while improved and r < rounds and time.time() < deadline - 2:
            improved = False
            r += 1
            for row_idx in range(n_rows):
                row = rows[row_idx]
                for pos in range(L - 1, 0, -1):
                    old_val = row[pos]
                    lower = row[pos - 1] + 1
                    if lower >= old_val:
                        continue
                    old = diffs_at(row, pos)
                    found = False
                    for nv in range(lower, old_val):
                        if pos < L - 1 and nv >= row[pos + 1]:
                            break
                        nd = set()
                        for i in range(L):
                            if i < pos:
                                nd.add(nv - row[i])
                            elif i > pos:
                                nd.add(row[i] - nv)
                        if nd & (used - old):
                            continue
                        for d in old:
                            used.discard(d)
                        for d in nd:
                            used.add(d)
                        row[pos] = nv
                        found = True
                        break
                    if found:
                        ns = max(max(rr) for rr in rows)
                        if ns < scope:
                            scope = ns
                        improved = True
        return rows, scope

    def local_improve(rows, seed, max_iters=2000):
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        used = build_used(rows)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            pos = rng.randint(1, L - 1)
            old_val = row[pos]
            if old_val <= 1:
                continue
            old = diffs_at(row, pos)
            for d in old:
                used.discard(d)
            row.pop(pos)
            lower = row[pos - 1] + 1 if pos > 0 else 1
            found = False
            for nv in range(lower, old_val):
                if pos < L - 1 and nv >= row[pos + 1]:
                    break
                nd = set()
                for i in range(L):
                    if i < pos:
                        nd.add(nv - row[i])
                    elif i > pos:
                        nd.add(row[i] - nv)
                if nd & used:
                    continue
                row.insert(pos, nv)
                for d in nd:
                    used.add(d)
                found = True
                break
            if found:
                ns = max(max(rr) for rr in rows)
                if ns < scope:
                    scope = ns
            else:
                row.insert(pos, old_val)
                for d in old:
                    used.add(d)
        return rows, scope

    # ---------- bitset backtracking (targeted scope reduction) ----------
    def backtracking_search(max_scope, seed, time_limit=1.5):
        rng = random.Random(seed)
        used_diffs = [False] * (max_scope + 1)
        rows = []
        t_start = time.time()

        def build_row(pos, last_val, row_so_far):
            nonlocal used_diffs
            if time.time() - t_start > time_limit:
                return None
            if pos == L:
                return list(row_so_far)
            max_x = max_scope - (L - pos - 1)
            if max_x <= last_val:
                return None
            valid = []
            for x in range(last_val + 1, max_x + 1):
                ok = True
                for a in row_so_far:
                    if used_diffs[x - a]:
                        ok = False
                        break
                if ok:
                    valid.append(x)
                if len(valid) >= 60:
                    break
            if not valid:
                return None
            n_take = min(16, len(valid))
            window = valid[:n_take]
            if len(valid) > n_take and rng.random() < 0.4:
                window.extend(rng.sample(valid[n_take:], min(6, len(valid) - n_take)))
            rng.shuffle(window)
            for x in window:
                nd = []
                for a in row_so_far:
                    d = x - a
                    nd.append(d)
                    used_diffs[d] = True
                res = build_row(pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res
                for d in nd:
                    used_diffs[d] = False
            return None

        for _ in range(n_rows):
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            for i in range(L):
                for j in range(i + 1, L):
                    used_diffs[row[j] - row[i]] = True
        return rows

    # ---------- simulated annealing ----------
    def simulated_annealing(rows, seed, time_limit=15.0):
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        used = build_used(rows)
        t_start = time.time()
        current_temp = max(1.0, scope * 0.1)
        cooling_rate = 0.995
        min_temp = 0.01
        best_rows = [list(r) for r in rows]
        best_scope = scope

        while current_temp > min_temp and time.time() - t_start < time_limit:
            if time.time() > deadline - 5:
                break
            if rng.random() < 0.7:
                row_idx = rng.randint(0, n_rows - 1)
                row = rows[row_idx]
                pos = rng.randint(1, L - 1)
                old_val = row[pos]
                if old_val <= 1:
                    continue
                old = diffs_at(row, pos)
                for d in old:
                    used.discard(d)
                row.pop(pos)
                lower = row[pos - 1] + 1 if pos > 0 else 1
                upper = min(old_val + rng.randint(-3, 3), scope)
                found = False
                for nv in range(lower, upper + 1):
                    if pos < L - 1 and nv >= row[pos + 1]:
                        break
                    nd = set()
                    for i in range(L):
                        if i < pos:
                            nd.add(nv - row[i])
                        elif i > pos:
                            nd.add(row[i] - nv)
                    if nd & used:
                        continue
                    row.insert(pos, nv)
                    for d in nd:
                        used.add(d)
                    found = True
                    break
                if not found:
                    row.insert(pos, old_val)
                    for d in old:
                        used.add(d)
                    continue
                ns = max(max(rr) for rr in rows)
                delta = ns - scope
                if delta <= 0 or rng.random() < math.exp(-delta / max(current_temp, 1e-6)):
                    scope = ns
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(rr) for rr in rows]
                else:
                    nd2 = diffs_at(row, pos)
                    for d in nd2:
                        used.discard(d)
                    row.pop(pos)
                    row.insert(pos, old_val)
                    for d in old:
                        used.add(d)
            else:
                row_idx = rng.randint(0, n_rows - 1)
                old_row = list(rows[row_idx])
                for i in range(L):
                    for j in range(i + 1, L):
                        used.discard(old_row[j] - old_row[i])
                new_row = [0]
                x = 0
                ok = True
                while len(new_row) < L:
                    x += 1
                    if x > scope:
                        ok = False
                        break
                    nd = {x - a for a in new_row}
                    if nd & used:
                        continue
                    new_row.append(x)
                    used |= nd
                if not ok:
                    for i in range(L):
                        for j in range(i + 1, L):
                            used.add(old_row[j] - old_row[i])
                    continue
                rows[row_idx] = new_row
                ns = max(max(rr) for rr in rows)
                delta = ns - scope
                if delta <= 0 or rng.random() < math.exp(-delta / max(current_temp, 1e-6)):
                    scope = ns
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(rr) for rr in rows]
                else:
                    for i in range(L):
                        for j in range(i + 1, L):
                            used.discard(new_row[j] - new_row[i])
                    rows[row_idx] = old_row
                    for i in range(L):
                        for j in range(i + 1, L):
                            used.add(old_row[j] - old_row[i])
            current_temp *= cooling_rate
        return best_rows, best_scope

    # ---------- high-precision final polish (extra work enabled by faster search) ----------
    def polish(rows):
        rows, s = compact_pass(rows, 7, rounds=2000)
        rows, s2 = local_improve(rows, 91, max_iters=1200)
        if s2 < s:
            s = s2
        rows, s3 = compact_pass(rows, 92, rounds=2000)
        if s3 < s:
            s = s3
        return rows, s

    # ---------- main pipeline ----------
    rows = greedy_baseline()
    if rows:
        best_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]
        best_rows, best_scope = polish(best_rows)

    # Phase: targeted backtracking scope reduction (primary engine)
    target = best_scope - 1
    attempt = 0
    while target > lb and time.time() < deadline - 30:
        found = False
        for _ in range(25):
            if time.time() > deadline - 30:
                break
            sol = backtracking_search(target, 100000 + attempt, time_limit=1.5)
            if sol:
                ns = max(max(r) for r in sol)
                if ns < best_scope:
                    best_scope = ns
                    best_rows = [list(r) for r in sol]
                    best_rows, best_scope = polish(best_rows)
                    found = True
                    break
            attempt += 1
        if found:
            target = best_scope - 1
        else:
            target -= 1
            if target <= lb:
                break

    # Phase: simulated annealing
    if best_rows and time.time() < deadline - 15:
        sa_rows, sa_scope = simulated_annealing(best_rows, 88888, time_limit=12.0)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
            best_rows, best_scope = polish(best_rows)

    # Phase: more SA with different seeds
    if best_rows and time.time() < deadline - 10:
        for off in [11111, 22222, 33333]:
            if time.time() > deadline - 5:
                break
            sa_rows, sa_scope = simulated_annealing(best_rows, 88888 + off, time_limit=5.0)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows
                best_rows, best_scope = polish(best_rows)

    # Phase: final high-precision polish
    if best_rows and time.time() < deadline - 3:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            cr, cs = compact_pass(best_rows, 555 + _, rounds=2000)
            if cs < best_scope:
                best_scope = cs
                best_rows = cr
            else:
                break

    return best_rows


if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
