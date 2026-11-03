import time
import random

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 112.0
    row_len = k + 1
    total_diffs_per_row = k * (k + 1) // 2

    def row_diffs(row):
        d = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d.add(row[j] - ri)
        return d

    def greedy_row(used):
        row = [0]; x = 0
        while len(row) < row_len:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x); used |= new
        return row

    def find_row(used, best_max_init, S_cap, node_budget, dl, rng):
        best = [None]; best_max = [best_max_init]; nodes = [0]
        def dfs(row, rdiffs, last):
            if time.time() > dl or nodes[0] >= node_budget:
                return
            nodes[0] += 1
            if len(row) == row_len:
                m = row[-1]
                if m < best_max[0]:
                    best_max[0] = m; best[0] = row[:]
                return
            hi = min(best_max[0] - 1, S_cap); lo = last + 1
            if lo > hi:
                return
            for x in range(lo, hi + 1):
                new = {x - a for a in row}
                if new & used or new & rdiffs:
                    continue
                row.append(x); rdiffs |= new
                dfs(row, rdiffs, x)
                rdiffs -= new; row.pop()
                if nodes[0] >= node_budget or time.time() > dl:
                    return
        dfs([0], set(), 0)
        if best[0] is None:
            return None
        return best[0], row_diffs(best[0])

    def scope_of(rows):
        return max(r[-1] for r in rows)

    # CSP with forward checking for initial construction
    def csp_build_rows(target_scope, time_limit, seed=42):
        rng = random.Random(seed)
        start_time = time.time()
        used = set()
        rows = []
        
        def build_one_row():
            row = [0]
            remaining = k
            
            def dfs_row(current_row, rem):
                if time.time() - start_time > time_limit:
                    return None
                if rem == 0:
                    return list(current_row)
                
                last = current_row[-1]
                candidates = []
                for v in range(last + 1, target_scope + 1):
                    new_diffs = [v - a for a in current_row]
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                    if any(d in used for d in new_diffs):
                        continue
                    candidates.append((v, new_diffs))
                
                if not candidates:
                    return None
                
                rng.shuffle(candidates)
                
                for v, new_diffs in candidates:
                    old_used = set(used)
                    used.update(new_diffs)
                    result = dfs_row(current_row + [v], rem - 1)
                    if result is not None:
                        return result
                    used.clear()
                    used.update(old_used)
                
                return None
            
            return dfs_row(row, k)
        
        for _ in range(n_rows):
            row = build_one_row()
            if row is None:
                return None
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
            rows.append(row)
        
        return rows

    # Incremental SA with diff counting
    def build_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def get_collisions(counts):
        coll = 0
        for d, c in counts.items():
            if c > 1:
                coll += c - 1
        return coll

    def simulate_annealing(initial_rows, time_limit, best_max):
        rows = [r[:] for r in initial_rows]
        current_max = max(r[-1] for r in rows)
        
        counts = build_diff_counts(rows)
        current_collisions = get_collisions(counts)
        current_cost = current_collisions * 10000 + current_max
        
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max
        
        T = 50.0
        cooling = 0.9998
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            
            lower = rows[r_idx][i_idx - 1] + 1
            upper_limit = best_max + 30
            if i_idx < row_len - 1:
                upper = rows[r_idx][i_idx + 1] - 1
                upper = min(upper, upper_limit)
            else:
                upper = upper_limit
            
            if lower > upper:
                continue
            
            new_val = random.randint(lower, upper)
            if new_val == rows[r_idx][i_idx]:
                continue
            
            old_val = rows[r_idx][i_idx]
            
            delta_coll = 0
            for j in range(row_len):
                if j == i_idx:
                    continue
                old_d = abs(old_val - rows[r_idx][j])
                new_d = abs(new_val - rows[r_idx][j])
                if old_d != new_d:
                    oc = counts.get(old_d, 0)
                    if oc > 0:
                        if oc == 1:
                            del counts[old_d]
                        else:
                            counts[old_d] = oc - 1
                            delta_coll -= 1
                    nc = counts.get(new_d, 0)
                    if nc == 0:
                        counts[new_d] = 1
                    else:
                        counts[new_d] = nc + 1
                        delta_coll += 1
            
            if i_idx > 0 and new_val <= rows[r_idx][i_idx - 1]:
                for j in range(row_len):
                    if j == i_idx:
                        continue
                    old_d = abs(new_val - rows[r_idx][j])
                    new_d = abs(old_val - rows[r_idx][j])
                    if old_d != new_d:
                        oc = counts.get(old_d, 0)
                        if oc > 0:
                            if oc == 1:
                                del counts[old_d]
                            else:
                                counts[old_d] = oc - 1
                        nc = counts.get(new_d, 0)
                        if nc == 0:
                            counts[new_d] = 1
                        else:
                            counts[new_d] = nc + 1
                continue
            
            if i_idx < row_len - 1 and new_val >= rows[r_idx][i_idx + 1]:
                for j in range(row_len):
                    if j == i_idx:
                        continue
                    old_d = abs(new_val - rows[r_idx][j])
                    new_d = abs(old_val - rows[r_idx][j])
                    if old_d != new_d:
                        oc = counts.get(old_d, 0)
                        if oc > 0:
                            if oc == 1:
                                del counts[old_d]
                            else:
                                counts[old_d] = oc - 1
                        nc = counts.get(new_d, 0)
                        if nc == 0:
                            counts[new_d] = 1
                        else:
                            counts[new_d] = nc + 1
                continue
            
            rows[r_idx][i_idx] = new_val
            new_max = max(rows[r_idx][-1], best_local_max)
            new_cost = (current_collisions + delta_coll) * 10000 + new_max
            
            delta_cost = new_cost - current_cost
            if delta_cost <= 0 or random.random() < pow(2.71828, -delta_cost / max(T, 0.01)):
                current_collisions += delta_coll
                current_cost = new_cost
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = max(r[-1] for r in rows)
            else:
                for j in range(row_len):
                    if j == i_idx:
                        continue
                    old_d = abs(new_val - rows[r_idx][j])
                    new_d = abs(old_val - rows[r_idx][j])
                    if old_d != new_d:
                        oc = counts.get(old_d, 0)
                        if oc > 0:
                            if oc == 1:
                                del counts[old_d]
                            else:
                                counts[old_d] = oc - 1
                        nc = counts.get(new_d, 0)
                        if nc == 0:
                            counts[new_d] = 1
                        else:
                            counts[new_d] = nc + 1
                rows[r_idx][i_idx] = old_val
            
            T *= cooling
        
        return best_local_max, best_local_rows

    def run(seed, dl):
        rng = random.Random(seed)
        used = set(); rows = []
        
        # Try CSP first for better initial solution
        csp_start = time.time()
        csp_time = min(10.0, dl - time.time() - 5)
        if csp_time > 2:
            greedy_rows = []
            tmp_used = set()
            for _ in range(n_rows):
                row = greedy_row(tmp_used)
                tmp_used |= row_diffs(row)
                greedy_rows.append(row)
            greedy_max = max(r[-1] for r in greedy_rows)
            
            csp_result = csp_build_rows(greedy_max - 1, csp_time, seed)
            if csp_result is not None:
                rows = csp_result
                for r in rows:
                    used |= row_diffs(r)
            else:
                rows = greedy_rows
                for r in rows:
                    used |= row_diffs(r)
        else:
            for _ in range(n_rows):
                row = greedy_row(used)
                used |= row_diffs(row)
                rows.append(row)
        
        best_rows = [r[:] for r in rows]; best_scope = scope_of(rows)
        
        # Run SA to polish
        sa_time = dl - time.time() - 3
        if sa_time > 2:
            sa_max, sa_rows = simulate_annealing(rows, sa_time, best_scope)
            if sa_max < best_scope:
                best_scope = sa_max
                best_rows = [r[:] for r in sa_rows]
        
        return best_scope, best_rows

    try:
        overall = None; seed = 0; first = True
        while time.time() < DEADLINE - 5:
            if first:
                dl = min(DEADLINE - 3, time.time() + 60); first = False
            else:
                dl = min(DEADLINE - 3, time.time() + 15)
            sc, rows = run(seed, dl)
            if overall is None or sc < overall[0]:
                overall = (sc, rows)
            seed += 1
        if overall is not None:
            return overall[1]
    except Exception:
        pass
    # last-resort valid greedy
    used = set(); rows = []
    for _ in range(n_rows):
        row = greedy_row(used); used |= row_diffs(row); rows.append(row)
    return rows
