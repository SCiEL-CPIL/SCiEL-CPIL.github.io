import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_all = n_rows * total_diffs_per_row

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = max(r[-1] for r in best_rows)

    # CSP backtracking with forward checking to build rows within a target scope
    def csp_build_rows(target_scope, time_limit, seed=None):
        rng = random.Random(seed)
        start_time = time.time()
        used = set()
        rows = []
        
        def build_one_row():
            """Build one row with forward checking. Returns list or None."""
            row = [0]
            
            def dfs_row(current_row, remaining):
                if time.time() - start_time > time_limit:
                    return None
                if remaining == 0:
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
                
                rng.shuffle(candidates)
                
                for v, new_diffs in candidates:
                    old_used = set(used)
                    used.update(new_diffs)
                    result = dfs_row(current_row + [v], remaining - 1)
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

    deadline = time.time() + 115.0

    # Try CSP to get a tighter initial solution
    csp_start = time.time()
    csp_time = 15.0
    if time.time() + csp_time < deadline:
        for target in range(best_max, max(best_max - 20, 100) - 1, -1):
            if time.time() - csp_start > csp_time:
                break
            result = csp_build_rows(target, csp_time - (time.time() - csp_start), seed=42)
            if result is not None:
                rmax = max(r[-1] for r in result)
                if rmax < best_max:
                    best_max = rmax
                    best_rows = result
                break

    # Efficient diff tracking with numpy for speed
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

    def simulate_annealing(initial_rows, time_limit, initial_best_max):
        rows = [r[:] for r in initial_rows]
        current_max = max(r[-1] for r in rows)
        
        counts = build_diff_counts(rows)
        current_collisions = get_collisions(counts)
        current_cost = current_collisions * 10000 + current_max
        
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max
        
        # Better temperature schedule: start high, cool faster
        T = 100.0
        cooling = 0.99995
        start_time = time.time()
        steps = 0
        max_steps = 500000  # Safety cap
        
        while time.time() - start_time < time_limit and steps < max_steps:
            # Choose move type: 70% single point, 20% swap two points in same row, 10% swap points between rows
            move_type = random.random()
            
            if move_type < 0.7:
                # Single point move
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                
                lower = rows[r_idx][i_idx - 1] + 1
                upper_limit = best_max + 20
                if i_idx < row_len - 1:
                    upper = min(rows[r_idx][i_idx + 1] - 1, upper_limit)
                else:
                    upper = upper_limit
                
                if lower > upper:
                    steps += 1
                    T *= cooling
                    continue
                
                new_val = random.randint(lower, upper)
                if new_val == rows[r_idx][i_idx]:
                    steps += 1
                    T *= cooling
                    continue
                
                old_val = rows[r_idx][i_idx]
                
                # Compute delta collisions
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
                
                # Validate
                valid = True
                if i_idx > 0 and new_val <= rows[r_idx][i_idx - 1]:
                    valid = False
                if i_idx < row_len - 1 and new_val >= rows[r_idx][i_idx + 1]:
                    valid = False
                
                if not valid:
                    # Revert
                    for j in range(row_len):
                        if j == i_idx:
                            continue
                        old_d = abs(old_val - rows[r_idx][j])
                        new_d = abs(new_val - rows[r_idx][j])
                        if old_d != new_d:
                            nc = counts.get(new_d, 0)
                            if nc == 1:
                                del counts[new_d]
                            else:
                                counts[new_d] = nc - 1
                            oc = counts.get(old_d, 0)
                            if oc == 0:
                                counts[old_d] = 1
                            else:
                                counts[old_d] = oc + 1
                    steps += 1
                    T *= cooling
                    continue
                
                rows[r_idx][i_idx] = new_val
                new_max = max(r[-1] for r in rows)
                new_collisions = current_collisions + delta_coll
                new_cost = new_collisions * 10000 + new_max
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    current_collisions = new_collisions
                    current_max = new_max
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_max = new_max
                        best_local_rows = [r[:] for r in rows]
                else:
                    delta = new_cost - current_cost
                    if delta == 0:
                        if random.random() < 0.5:
                            current_cost = new_cost
                            current_collisions = new_collisions
                            current_max = new_max
                        else:
                            rows[r_idx][i_idx] = old_val
                            for j in range(row_len):
                                if j == i_idx:
                                    continue
                                old_d = abs(old_val - rows[r_idx][j])
                                new_d = abs(new_val - rows[r_idx][j])
                                if old_d != new_d:
                                    nc = counts.get(new_d, 0)
                                    if nc == 1:
                                        del counts[new_d]
                                    else:
                                        counts[new_d] = nc - 1
                                    oc = counts.get(old_d, 0)
                                    if oc == 0:
                                        counts[old_d] = 1
                                    else:
                                        counts[old_d] = oc + 1
                    else:
                        p = math.exp(-delta / T) if delta < 8000 else 0.0
                        if random.random() < p:
                            current_cost = new_cost
                            current_collisions = new_collisions
                            current_max = new_max
                        else:
                            rows[r_idx][i_idx] = old_val
                            for j in range(row_len):
                                if j == i_idx:
                                    continue
                                old_d = abs(old_val - rows[r_idx][j])
                                new_d = abs(new_val - rows[r_idx][j])
                                if old_d != new_d:
                                    nc = counts.get(new_d, 0)
                                    if nc == 1:
                                        del counts[new_d]
                                    else:
                                        counts[new_d] = nc - 1
                                    oc = counts.get(old_d, 0)
                                    if oc == 0:
                                        counts[old_d] = 1
                                    else:
                                        counts[old_d] = oc + 1
            
            elif move_type < 0.9:
                # Swap two points in the same row
                r_idx = random.randint(0, n_rows - 1)
                i_idx, j_idx = sorted(random.sample(range(1, row_len), 2))
                
                old_i = rows[r_idx][i_idx]
                old_j = rows[r_idx][j_idx]
                
                # Try swapping
                new_i = old_j
                new_j = old_i
                
                # Check validity
                valid = True
                if i_idx > 0 and new_i <= rows[r_idx][i_idx - 1]:
                    valid = False
                if i_idx < row_len - 1 and new_i >= rows[r_idx][i_idx + 1]:
                    valid = False
                if j_idx > 0 and new_j <= rows[r_idx][j_idx - 1]:
                    valid = False
                if j_idx < row_len - 1 and new_j >= rows[r_idx][j_idx + 1]:
                    valid = False
                
                if not valid:
                    steps += 1
                    T *= cooling
                    continue
                
                # Compute delta
                delta_coll = 0
                # Remove old diffs
                for idx in [i_idx, j_idx]:
                    old_val = rows[r_idx][idx]
                    for other in range(row_len):
                        if other == idx:
                            continue
                        d = abs(old_val - rows[r_idx][other])
                        oc = counts.get(d, 0)
                        if oc == 1:
                            del counts[d]
                        else:
                            counts[d] = oc - 1
                            delta_coll -= 1
                
                # Add new diffs
                rows[r_idx][i_idx] = new_i
                rows[r_idx][j_idx] = new_j
                for idx in [i_idx, j_idx]:
                    new_val = rows[r_idx][idx]
                    for other in range(row_len):
                        if other == idx:
                            continue
                        d = abs(new_val - rows[r_idx][other])
                        nc = counts.get(d, 0)
                        if nc == 0:
                            counts[d] = 1
                        else:
                            counts[d] = nc + 1
                            delta_coll += 1
                
                new_max = max(r[-1] for r in rows)
                new_collisions = current_collisions + delta_coll
                new_cost = new_collisions * 10000 + new_max
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    current_collisions = new_collisions
                    current_max = new_max
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_max = new_max
                        best_local_rows = [r[:] for r in rows]
                else:
                    delta = new_cost - current_cost
                    if delta == 0:
                        if random.random() < 0.5:
                            current_cost = new_cost
                            current_collisions = new_collisions
                            current_max = new_max
                        else:
                            # Revert
                            rows[r_idx][i_idx] = old_i
                            rows[r_idx][j_idx] = old_j
                            # Remove new diffs
                            for idx in [i_idx, j_idx]:
                                new_val = rows[r_idx][idx]
                                for other in range(row_len):
                                    if other == idx:
                                        continue
                                    d = abs(new_val - rows[r_idx][other])
                                    nc = counts.get(d, 0)
                                    if nc == 1:
                                        del counts[d]
                                    else:
                                        counts[d] = nc - 1
                            # Add old diffs
                            for idx in [i_idx, j_idx]:
                                old_val = rows[r_idx][idx]
                                for other in range(row_len):
                                    if other == idx:
                                        continue
                                    d = abs(old_val - rows[r_idx][other])
                                    oc = counts.get(d, 0)
                                    if oc == 0:
                                        counts[d] = 1
                                    else:
                                        counts[d] = oc + 1
                    else:
                        p = math.exp(-delta / T) if delta < 8000 else 0.0
                        if random.random() < p:
                            current_cost = new_cost
                            current_collisions = new_collisions
                            current_max = new_max
                        else:
                            # Revert
                            rows[r_idx][i_idx] = old_i
                            rows[r_idx][j_idx] = old_j
                            for idx in [i_idx, j_idx]:
                                new_val = rows[r_idx][idx]
                                for other in range(row_len):
                                    if other == idx:
                                        continue
                                    d = abs(new_val - rows[r_idx][other])
                                    nc = counts.get(d, 0)
                                    if nc == 1:
                                        del counts[d]
                                    else:
                                        counts[d] = nc - 1
                            for idx in [i_idx, j_idx]:
                                old_val = rows[r_idx][idx]
                                for other in range(row_len):
                                    if other == idx:
                                        continue
                                    d = abs(old_val - rows[r_idx][other])
                                    oc = counts.get(d, 0)
                                    if oc == 0:
                                        counts[d] = 1
                                    else:
                                        counts[d] = oc + 1
            
            else:
                # Swap points between two different rows
                r1, r2 = random.sample(range(n_rows), 2)
                i1 = random.randint(1, row_len - 1)
                i2 = random.randint(1, row_len - 1)
                
                old1 = rows[r1][i1]
                old2 = rows[r2][i2]
                
                new1 = old2
                new2 = old1
                
                # Check validity
                valid = True
                if i1 > 0 and new1 <= rows[r1][i1 - 1]:
                    valid = False
                if i1 < row_len - 1 and new1 >= rows[r1][i1 + 1]:
                    valid = False
                if i2 > 0 and new2 <= rows[r2][i2 - 1]:
                    valid = False
                if i2 < row_len - 1 and new2 >= rows[r2][i2 + 1]:
                    valid = False
                
                if not valid:
                    steps += 1
                    T *= cooling
                    continue
                
                # Compute delta
                delta_coll = 0
                # Remove old diffs from both rows
                for r_idx, i_idx in [(r1, i1), (r2, i2)]:
                    old_val = rows[r_idx][i_idx]
                    for other in range(row_len):
                        if other == i_idx:
                            continue
                        d = abs(old_val - rows[r_idx][other])
                        oc = counts.get(d, 0)
                        if oc == 1:
                            del counts[d]
                        else:
                            counts[d] = oc - 1
                            delta_coll -= 1
                
                # Apply and add new diffs
                rows[r1][i1] = new1
                rows[r2][i2] = new2
                for r_idx, i_idx in [(r1, i1), (r2, i2)]:
                    new_val = rows[r_idx][i_idx]
                    for other in range(row_len):
                        if other == i_idx:
                            continue
                        d = abs(new_val - rows[r_idx][other])
                        nc = counts.get(d, 0)
                        if nc == 0:
                            counts[d] = 1
                        else:
                            counts[d] = nc + 1
                            delta_coll += 1
                
                new_max = max(r[-1] for r in rows)
                new_collisions = current_collisions + delta_coll
                new_cost = new_collisions * 10000 + new_max
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    current_collisions = new_collisions
                    current_max = new_max
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_max = new_max
                        best_local_rows = [r[:] for r in rows]
                else:
                    delta = new_cost - current_cost
                    if delta == 0:
                        if random.random() < 0.5:
                            current_cost = new_cost
                            current_collisions = new_collisions
                            current_max = new_max
                        else:
                            # Revert
                            rows[r1][i1] = old1
                            rows[r2][i2] = old2
                            for r_idx, i_idx in [(r1, i1), (r2, i2)]:
                                new_val = rows[r_idx][i_idx]
                                for other in range(row_len):
                                    if other == i_idx:
                                        continue
                                    d = abs(new_val - rows[r_idx][other])
                                    nc = counts.get(d, 0)
                                    if nc == 1:
                                        del counts[d]
                                    else:
                                        counts[d] = nc - 1
                            for r_idx, i_idx in [(r1, i1), (r2, i2)]:
                                old_val = rows[r_idx][i_idx]
                                for other in range(row_len):
                                    if other == i_idx:
                                        continue
                                    d = abs(old_val - rows[r_idx][other])
                                    oc = counts.get(d, 0)
                                    if oc == 0:
                                        counts[d] = 1
                                    else:
                                        counts[d] = oc + 1
                    else:
                        p = math.exp(-delta / T) if delta < 8000 else 0.0
                        if random.random() < p:
                            current_cost = new_cost
                            current_collisions = new_collisions
                            current_max = new_max
                        else:
                            # Revert
                            rows[r1][i1] = old1
                            rows[r2][i2] = old2
                            for r_idx, i_idx in [(r1, i1), (r2, i2)]:
                                new_val = rows[r_idx][i_idx]
                                for other in range(row_len):
                                    if other == i_idx:
                                        continue
                                    d = abs(new_val - rows[r_idx][other])
                                    nc = counts.get(d, 0)
                                    if nc == 1:
                                        del counts[d]
                                    else:
                                        counts[d] = nc - 1
                            for r_idx, i_idx in [(r1, i1), (r2, i2)]:
                                old_val = rows[r_idx][i_idx]
                                for other in range(row_len):
                                    if other == i_idx:
                                        continue
                                    d = abs(old_val - rows[r_idx][other])
                                    oc = counts.get(d, 0)
                                    if oc == 0:
                                        counts[d] = 1
                                    else:
                                        counts[d] = oc + 1
            
            T *= cooling
            if T < 0.1:
                T = 0.1
            steps += 1
            
            # Verify consistency periodically
            if steps % 10000 == 0:
                actual_coll = get_collisions(counts)
                if actual_coll != current_collisions:
                    counts = build_diff_counts(rows)
                    current_collisions = get_collisions(counts)
                    current_max = max(r[-1] for r in rows)
                    current_cost = current_collisions * 10000 + current_max
        
        return best_local_rows, best_local_max, best_local_cost

    # Final polish: targeted local search to reduce max entry
    def polish_rows(rows, time_limit):
        """Aggressive local search to reduce the maximum entry."""
        rows = [r[:] for r in rows]
        counts = build_diff_counts(rows)
        current_collisions = get_collisions(counts)
        current_max = max(r[-1] for r in rows)
        best_max = current_max
        best_rows = [r[:] for r in rows]
        
        start_time = time.time()
        T = 20.0
        cooling = 0.9999
        
        while time.time() - start_time < time_limit:
            # Try to reduce the max entry
            max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            
            # Try moving the last element of the max row
            r_idx = max_row_idx
            i_idx = row_len - 1
            
            lower = rows[r_idx][i_idx - 1] + 1
            upper = rows[r_idx][i_idx] - 1
            
            if lower <= upper:
                new_val = random.randint(lower, upper)
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
                
                rows[r_idx][i_idx] = new_val
                new_max = max(r[-1] for r in rows)
                new_collisions = current_collisions + delta_coll
                new_cost = new_collisions * 10000 + new_max
                
                if new_cost < current_collisions * 10000 + current_max:
                    current_collisions = new_collisions
                    current_max = new_max
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [r[:] for r in rows]
                else:
                    delta = new_cost - (current_collisions * 10000 + current_max)
                    if delta == 0 and random.random() < 0.5:
                        current_collisions = new_collisions
                        current_max = new_max
                    else:
                        p = math.exp(-delta / T) if delta < 5000 else 0.0
                        if random.random() < p:
                            current_collisions = new_collisions
                            current_max = new_max
                        else:
                            rows[r_idx][i_idx] = old_val
                            for j in range(row_len):
                                if j == i_idx:
                                    continue
                                old_d = abs(old_val - rows[r_idx][j])
                                new_d = abs(new_val - rows[r_idx][j])
                                if old_d != new_d:
                                    nc = counts.get(new_d, 0)
                                    if nc == 1:
                                        del counts[new_d]
                                    else:
                                        counts[new_d] = nc - 1
                                    oc = counts.get(old_d, 0)
                                    if oc == 0:
                                        counts[old_d] = 1
                                    else:
                                        counts[old_d] = oc + 1
            
            T *= cooling
            if T < 0.05:
                T = 0.05
        
        return best_rows, best_max

    # Main SA loop
    while time.time() < deadline - 5.0:
        if best_rows is not None and all(r[0] == 0 and len(r) == row_len for r in best_rows):
            all_diffs = []
            for r in best_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        all_diffs.append(r[j] - r[i])
            if len(set(all_diffs)) == len(all_diffs):
                init_rows = best_rows
            else:
                init_rows = []
                for _ in range(n_rows):
                    pts = random.sample(range(1, best_max + 1), k)
                    pts.sort()
                    init_rows.append([0] + pts)
        else:
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, best_max + 1), k)
                pts.sort()
                init_rows.append([0] + pts)
        
        remaining = deadline - time.time() - 5.0
        sa_time = min(remaining, 10.0)
        if sa_time <= 0:
            break
        
        local_rows, local_max, local_cost = simulate_annealing(init_rows, sa_time, best_max)
        
        if local_cost < 10000:
            if local_max < best_max:
                best_max = local_max
                best_rows = local_rows
    
    # Final polish if time permits
    if time.time() < deadline - 3.0 and best_rows is not None:
        polished_rows, polished_max = polish_rows(best_rows, deadline - time.time() - 3.0)
        if polished_max < best_max:
            best_max = polished_max
            best_rows = polished_rows

    return best_rows
