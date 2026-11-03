import time
import random
import math

def construct(n_rows, k):
    """
    Constructs a difference triangle set (rulers) with n_rows rows and k marks per row (excluding 0).
    Returns a list of rows, each a list of k+1 integers.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]
    
    start_time = time.time()
    deadline = start_time + 130.0
    m = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    min_scope = total_diffs

    def time_left():
        return deadline - time.time()

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

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    def gen_row(used, k_val, randomize, p_smallest=0.8, window=4, rng=None, cap=500):
        """Generate one row of k_val+1 entries starting at 0, avoiding used diffs."""
        if rng is None:
            rng = random
        row = [0]
        cur = set()
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            forbidden = used | cur
            while x < cap:
                new_d = {x - a for a in row}
                if not (new_d & forbidden):
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            if not candidates:
                return None
            if randomize:
                if rng.random() < p_smallest:
                    nxt = candidates[0]
                else:
                    nxt = rng.choice(candidates)
            else:
                nxt = candidates[0]
            row.append(nxt)
            cur |= {nxt - a for a in row[:-1]}
        return row

    def tighten(rows, max_iter=200):
        """Pull each element (right to left, row by row) down to its smallest valid value."""
        rows = [list(r) for r in rows]
        for _ in range(max_iter):
            if time_left() < 15:
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

    def greedy_initial(rng):
        used = set()
        rows = []
        prev_first = 0
        for r in range(n_rows):
            row = None
            attempts = 0
            while row is None and attempts < 100:
                attempts += 1
                row = gen_row(used, k, randomize=True, p_smallest=0.85,
                              window=3, rng=rng, cap=400)
                if row is not None:
                    if r > 0 and row[1] <= prev_first:
                        row = None
                        continue
            if row is None:
                row = gen_row(used, k, randomize=False, rng=rng, cap=500)
            rows.append(row)
            prev_first = row[1]
            used |= row_diffs(row)
        return rows

    def anneal(rows, rng, max_iter=50000):
        rows = [list(r) for r in rows]
        used = set()
        for r in rows:
            used |= row_diffs(r)
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        T = 10.0
        cooling = 0.9995
        min_T = 0.1

        for it in range(max_iter):
            if time_left() < 15:
                break
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = row_diffs(old_row)
            used -= old_diffs

            p_smallest = 0.5 + 0.4 * rng.random()
            window = 2 + rng.randint(0, 3)
            cap = 200 + rng.randint(0, 200)
            new_row = gen_row(used, k, randomize=True, p_smallest=p_smallest,
                              window=window, rng=rng, cap=cap)

            if new_row is None:
                used |= old_diffs
                continue

            new_diffs = row_diffs(new_row)
            other_max = 0
            for i in range(n_rows):
                if i != r_idx:
                    other_max = max(other_max, rows[i][-1])
            new_global = max(other_max, new_row[-1])
            delta = new_global - current_max

            if delta < 0:
                rows[r_idx] = new_row
                used |= new_diffs
                current_max = new_global
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            else:
                if delta == 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / max(T, 0.01))
                if rng.random() < prob:
                    rows[r_idx] = new_row
                    used |= new_diffs
                    current_max = new_global
                else:
                    used |= old_diffs

            T = max(min_T, T * cooling)

        return best_rows

    def local_search(rows, rng, max_iter=5000):
        rows = [list(r) for r in rows]
        used = set()
        for r in rows:
            used |= row_diffs(r)
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        for it in range(max_iter):
            if time_left() < 10:
                break
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, len(rows[r_idx]) - 1)
            old_val = rows[r_idx][pos]
            if old_val <= rows[r_idx][pos - 1] + 1:
                continue
            
            new_val = rows[r_idx][pos - 1] + 1
            if new_val >= old_val:
                continue
            
            cand = rows[r_idx][:pos] + [new_val] + rows[r_idx][pos + 1:]
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
                if idx == r_idx:
                    continue
                od = row_diffs(rows[idx])
                if idiffs & od:
                    ok = False
                    break
            if ok:
                old_diffs = row_diffs(rows[r_idx])
                used -= old_diffs
                rows[r_idx][pos] = new_val
                used |= idiffs
                new_max = scope_of(rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in rows]

        return best_rows

    # Main search
    rng = random.Random(42)
    best_rows = None
    best_scope = float('inf')

    # Phase 1: Greedy initialization
    rows = greedy_initial(rng)
    if is_valid(rows):
        rows = tighten(rows)
        s = scope_of(rows)
        if s < best_scope:
            best_scope = s
            best_rows = [list(r) for r in rows]

    # Phase 2: Annealing from greedy
    if time_left() > 30:
        annealed = anneal(rows, rng, max_iter=30000)
        if is_valid(annealed):
            annealed = tighten(annealed)
            s = scope_of(annealed)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in annealed]

    # Phase 3: Local search
    if time_left() > 15:
        ls_rows = local_search(annealed if time_left() > 20 else rows, rng, max_iter=3000)
        if is_valid(ls_rows):
            ls_rows = tighten(ls_rows)
            s = scope_of(ls_rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in ls_rows]

    # Phase 4: Multiple random restarts
    restarts = 0
    while time_left() > 20:
        restarts += 1
        rows2 = greedy_initial(rng)
        if not is_valid(rows2):
            continue
        rows2 = tighten(rows2)
        s2 = scope_of(rows2)
        if s2 < best_scope:
            best_scope = s2
            best_rows = [list(r) for r in rows2]
        
        if time_left() > 25:
            annealed2 = anneal(rows2, rng, max_iter=15000)
            if is_valid(annealed2):
                annealed2 = tighten(annealed2)
                s3 = scope_of(annealed2)
                if s3 < best_scope:
                    best_scope = s3
                    best_rows = [list(r) for r in annealed2]

    # Final tightening
    if best_rows is not None:
        best_rows = tighten(best_rows, max_iter=50)
        if is_valid(best_rows):
            return best_rows
        else:
            fallback = greedy_initial(rng)
            if is_valid(fallback):
                return fallback
            else:
                used = set()
                rows = []
                for _ in range(n_rows):
                    row = gen_row(used, k, randomize=False, cap=1000)
                    if row is None:
                        row = [0]
                        x = 0
                        while len(row) < k + 1:
                            x += 1
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used:
                                row.append(x)
                                used |= new_diffs
                    else:
                        used |= row_diffs(row)
                    rows.append(row)
                return rows
    else:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = gen_row(used, k, randomize=False, cap=1000)
            if row is None:
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
            else:
                used |= row_diffs(row)
            rows.append(row)
        return rows
