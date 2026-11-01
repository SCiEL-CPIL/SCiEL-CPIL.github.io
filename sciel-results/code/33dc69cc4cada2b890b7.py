import time
import random

def construct(n_rows, k):
    random.seed(42)
    num_points = k + 1
    start_time = time.time()
    time_limit = 110.0
    
    total_pairs_per_row = num_points * (num_points - 1) // 2
    total_pairs = n_rows * total_pairs_per_row

    def get_cost(rows):
        diff_count = {}
        max_scope = 0
        for row in rows:
            if row[-1] > max_scope:
                max_scope = row[-1]
            for i in range(num_points):
                ri = row[i]
                for j in range(i + 1, num_points):
                    d = row[j] - ri
                    diff_count[d] = diff_count.get(d, 0) + 1
        
        unique_diffs = len(diff_count)
        collisions = total_pairs - unique_diffs
        return collisions, max_scope

    def initial_solution():
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            while len(row) < num_points:
                found = False
                last_val = row[-1]
                offset = random.randint(0, 2)
                for x in range(last_val + 1 + offset, last_val + 1 + 50):
                    ok = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        found = True
                        break
                if not found:
                    row.append(last_val + 1)
            rows.append(row)
        return rows

    def kick(rows):
        r = random.randint(0, n_rows - 1)
        other_diffs = set()
        for i in range(n_rows):
            if i == r: continue
            row_i = rows[i]
            for a in range(num_points):
                ra = row_i[a]
                for b in range(a + 1, num_points):
                    other_diffs.add(row_i[b] - ra)
        
        new_row = [0]
        success = False
        for _ in range(10):
            candidate = [0]
            while len(candidate) < num_points:
                last = candidate[-1]
                found = False
                offset = random.randint(0, 3)
                for x in range(last + 1 + offset, last + 1 + 80):
                    ok = True
                    for a in candidate:
                        d = x - a
                        if d in other_diffs:
                            ok = False
                            break
                    if ok:
                        candidate.append(x)
                        found = True
                        break
                if not found:
                    break
            if len(candidate) == num_points:
                new_row = candidate
                success = True
                break
        
        if success:
            rows[r] = new_row
        else:
            for p in range(1, num_points):
                if random.random() < 0.5:
                    rows[r][p] += random.randint(-2, 2)
            for p in range(1, num_points):
                if rows[r][p] <= rows[r][p-1]:
                    rows[r][p] = rows[r][p-1] + 1
        return rows

    def local_search(rows, deadline):
        current_cost = get_cost(rows)
        best_rows = [list(r) for r in rows]
        best_cost = current_cost
        
        while time.time() < deadline:
            improved = False
            
            for r in range(n_rows):
                if improved: break
                row = rows[r]
                for p in range(1, num_points):
                    if improved: break
                    old_val = row[p]
                    for delta in [1, -1, 2, -2, 3, -3, 4, -4]:
                        new_val = old_val + delta
                        if new_val <= row[p-1]:
                            continue
                        if p < num_points - 1 and new_val >= row[p+1]:
                            continue
                        if new_val < 0:
                            continue
                        
                        row[p] = new_val
                        new_cost = get_cost(rows)
                        
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_rows = [list(x) for x in rows]
                            improved = True
                            break
                        row[p] = old_val
            
            if not improved:
                r = random.randint(0, n_rows - 1)
                old_row = list(rows[r])
                
                other_diffs = set()
                for i in range(n_rows):
                    if i == r: continue
                    row_i = rows[i]
                    for a in range(num_points):
                        ra = row_i[a]
                        for b in range(a + 1, num_points):
                            other_diffs.add(row_i[b] - ra)
                
                new_row = [0]
                success = False
                for _ in range(20):
                    candidate = [0]
                    while len(candidate) < num_points:
                        last = candidate[-1]
                        found = False
                        offset = random.randint(0, 5)
                        for x in range(last + 1 + offset, last + 1 + 100):
                            ok = True
                            for a in candidate:
                                d = x - a
                                if d in other_diffs:
                                    ok = False
                                    break
                            if ok:
                                candidate.append(x)
                                found = True
                                break
                        if not found:
                            break
                    if len(candidate) == num_points:
                        new_row = candidate
                        success = True
                        break
                
                if success:
                    rows[r] = new_row
                    new_cost = get_cost(rows)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_rows = [list(x) for x in rows]
                        improved = True
                    else:
                        rows[r] = old_row
                else:
                    rows[r] = old_row
            
            if not improved:
                break
        
        return best_rows, best_cost

    current_rows = initial_solution()
    current_rows, current_cost = local_search(current_rows, time_limit)
    best_rows = [list(r) for r in current_rows]
    best_cost = current_cost
    
    while time.time() < time_limit:
        perturbed_rows = kick([list(r) for r in current_rows])
        perturbed_rows, new_cost = local_search(perturbed_rows, time_limit)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_rows = [list(r) for r in perturbed_rows]
            current_rows = [list(r) for r in perturbed_rows]
        else:
            current_rows = [list(r) for r in perturbed_rows]
            
    return best_rows
