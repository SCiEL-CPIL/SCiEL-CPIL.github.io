import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140
    
    def is_valid(rows):
        seen = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def row_max(rows):
        return max(max(r) for r in rows)
    
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    def get_row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}
    
    def get_all_diffs(rows):
        s = set()
        for r in rows:
            s |= get_row_diffs(r)
        return s
    
    # Initial solution
    best_rows = greedy()
    best_max = row_max(best_rows)
    
    # Local search
    def local_search(rows, time_limit):
        rows = [list(r) for r in rows]
        best = [list(r) for r in rows]
        best_m = row_max(rows)
        cur_m = best_m
        t = 300.0
        cooling = 0.9995
        t_end = 1.0
        
        while time.time() < time_limit and t > t_end:
            move_type = random.random()
            
            if move_type < 0.6:
                # Single position adjustment
                ri = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                old_val = rows[ri][pos]
                # Try a range of deltas
                delta = random.randint(-15, 15)
                if delta == 0:
                    t *= cooling
                    continue
                new_val = old_val + delta
                
                if new_val <= 0:
                    t *= cooling
                    continue
                if pos > 0 and new_val <= rows[ri][pos - 1]:
                    t *= cooling
                    continue
                if pos < k and new_val >= rows[ri][pos + 1]:
                    t *= cooling
                    continue
                
                rows[ri][pos] = new_val
                if not all(rows[ri][i] < rows[ri][i + 1] for i in range(k)):
                    rows[ri][pos] = old_val
                    t *= cooling
                    continue
                
                new_m = row_max(rows)
                if is_valid(rows):
                    if new_m < best_m:
                        best_m = new_m
                        best = [list(r) for r in rows]
                        cur_m = new_m
                        t = min(t, 100.0)
                    elif random.random() < min(1.0, (cur_m - new_m) / t if t > 0 else 0):
                        cur_m = new_m
                    else:
                        rows[ri][pos] = old_val
                else:
                    rows[ri][pos] = old_val
            
            elif move_type < 0.85:
                # Rebuild one row from scratch given other rows' constraints
                ri = random.randint(0, n_rows - 1)
                old_row = rows[ri][:]
                other_diffs = get_all_diffs(rows) - get_row_diffs(old_row)
                
                new_row = [0]
                x = 0
                max_x = best_m + 20  # Allow some flexibility
                while len(new_row) < k + 1 and x < max_x:
                    x += 1
                    new = {x - a for a in new_row}
                    if not new & other_diffs:
                        new_row.append(x)
                
                if len(new_row) == k + 1:
                    rows[ri] = new_row
                    new_m = row_max(rows)
                    if is_valid(rows):
                        if new_m < best_m:
                            best_m = new_m
                            best = [list(r) for r in rows]
                            cur_m = new_m
                            t = min(t, 100.0)
                        elif random.random() < min(1.0, (cur_m - new_m) / t if t > 0 else 0):
                            cur_m = new_m
                        else:
                            rows[ri] = old_row
                    else:
                        rows[ri] = old_row
            
            else:
                # Pairwise swap: swap two rows
                r1, r2 = random.sample(range(n_rows), 2)
                rows[r1], rows[r2] = rows[r2], rows[r1]
                new_m = row_max(rows)
                if is_valid(rows):
                    if new_m < best_m:
                        best_m = new_m
                        best = [list(r) for r in rows]
                        cur_m = new_m
                    elif random.random() < min(1.0, (cur_m - new_m) / t if t > 0 else 0):
                        cur_m = new_m
                    else:
                        rows[r1], rows[r2] = rows[r2], rows[r1]
                else:
                    rows[r1], rows[r2] = rows[r2], rows[r1]
            
            t *= cooling
        
        return best, best_m
    
    # Main local search
    if time.time() < deadline - 10:
        best_rows, best_max = local_search(best_rows, deadline - 10)
    
    # Multiple restarts
    restarts = 0
    while time.time() < deadline - 5 and restarts < 15:
        rows = greedy()
        m = row_max(rows)
        if m <= best_max:
            rows, m = local_search(rows, deadline - 5)
            if m < best_max:
                best_rows = rows
                best_max = m
        restarts += 1
    
    # Final polish
    if time.time() < deadline - 2:
        best_rows, best_max = local_search(best_rows, deadline - 2)
    
    return best_rows
