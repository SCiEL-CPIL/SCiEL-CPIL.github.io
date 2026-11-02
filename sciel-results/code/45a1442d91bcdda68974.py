import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    def generate_row(forbidden, k, max_limit, greediness=0.7):
        row = [0]
        used_in_row = set()
        for _ in range(k):
            found = False
            candidates = []
            for x in range(1, max_limit + 1):
                if x <= row[-1]:
                    continue
                new_diffs = {x - a for a in row}
                if new_diffs & used_in_row:
                    continue
                if new_diffs & forbidden:
                    continue
                candidates.append((x, new_diffs))
                
                if random.random() < greediness:
                    row.append(x)
                    used_in_row |= new_diffs
                    found = True
                    break
            if not found:
                if candidates:
                    idx = random.randint(0, min(4, len(candidates)-1))
                    x, new_diffs = candidates[idx]
                    row.append(x)
                    used_in_row |= new_diffs
                    found = True
                else:
                    return None
        return row

    best_rows = None
    best_m = float('inf')
    
    # Initialization
    restarts = 0
    while time.time() < deadline and best_rows is None:
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            limit = int(n_rows * k * (k+1) / 2 * 2.0) 
            row = generate_row(used, k, limit, greediness=0.8)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= get_diffs(row)
        if success:
            best_rows = rows
            best_m = max(max(r) for r in rows)
            restarts += 1
            if restarts >= 20:
                break
        restarts += 1
        
    if best_rows is None:
        # Fallback
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

    # Local Search
    current_rows = [r[:] for r in best_rows]
    current_used = set()
    for r in current_rows:
        current_used |= get_diffs(r)
    current_m = max(max(r) for r in current_rows)
    
    while time.time() < deadline:
        i = random.randint(0, n_rows - 1)
        forbidden = current_used - get_diffs(current_rows[i])
        
        improved = False
        row = current_rows[i]
        
        for j in range(k, 0, -1):
            old_val = row[j]
            min_val = row[j-1] + 1
            for new_val in range(old_val - 1, min_val - 1, -1):
                new_row = row[:]
                new_row[j] = new_val
                if new_row[j-1] >= new_row[j]:
                    continue
                if j < k and new_row[j] >= new_row[j+1]:
                    continue
                
                new_diffs = get_diffs(new_row)
                if not (new_diffs & forbidden):
                    current_used -= get_diffs(row)
                    current_used |= new_diffs
                    current_rows[i] = new_row
                    new_m = max(max(r) for r in current_rows)
                    if new_m < best_m:
                        best_m = new_m
                        best_rows = [r[:] for r in current_rows]
                    improved = True
                    break
        
        if not improved and random.random() < 0.05:
            current_row_max = max(row)
            new_row = generate_row(forbidden, k, current_row_max - 1, greediness=0.5)
            if new_row is not None:
                current_used -= get_diffs(row)
                current_used |= get_diffs(new_row)
                current_rows[i] = new_row
                new_m = max(max(r) for r in current_rows)
                if new_m < best_m:
                    best_m = new_m
                    best_rows = [r[:] for r in current_rows]

    return best_rows
