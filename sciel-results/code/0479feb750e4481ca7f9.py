import random
import time

def construct(n_rows, k):
    # 1. Generate an initial valid solution using the greedy method
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
        
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    def get_cost(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(list(get_diffs(r)))
        num_conflicts = len(all_diffs) - len(set(all_diffs))
        max_entry = max(max(r) for r in rows)
        return num_conflicts * 100000 + max_entry

    current_cost = get_cost(rows)
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in current_rows]
    best_cost = current_cost
    
    start_time = time.time()
    time_limit = 110
    temp = 1000
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit and temp > 1:
        i = random.randint(0, n_rows - 1)
        j = random.randint(1, k)
        
        old_val = current_rows[i][j]
        
        min_val = current_rows[i][j-1] + 1
        if j < k:
            max_val = current_rows[i][j+1] - 1
        else:
            max_val = old_val + 50 # Allow some expansion
            
        if min_val > max_val:
            max_val = min_val + 20
            
        new_val = random.randint(min_val, max_val)
        
        if new_val == old_val:
            continue
            
        current_rows[i][j] = new_val
        
        if (j > 0 and current_rows[i][j] <= current_rows[i][j-1]) or (j < k and current_rows[i][j] >= current_rows[i][j+1]):
            current_rows[i][j] = old_val
            continue
            
        new_cost = get_cost(current_rows)
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.718, -delta / temp):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [list(r) for r in current_rows]
        else:
            current_rows[i][j] = old_val
            
        temp *= cooling_rate
        
    return best_rows
