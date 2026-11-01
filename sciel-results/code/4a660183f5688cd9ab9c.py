import random
import math
import time

def construct(n_rows, k):
    row_len = k + 1
    
    def remove_point_diffs(row, idx):
        val = row[idx]
        diffs = set()
        for j, v in enumerate(row):
            if j != idx:
                diffs.add(abs(val - v))
        return diffs

    def greedy_init_randomized():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                # Find a set of valid candidates in a window
                candidates = []
                x = row[-1] + 1
                limit = row[-1] + 50
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Fallback to standard greedy if no candidates found in window
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            candidates.append(x)
                            break
                        x += 1
                            
                # Pick randomly from candidates to diversify
                x = random.choice(candidates)
                new_diffs = {x - a for a in row}
                row.append(x)
                used.update(new_diffs)
            rows.append(row)
        return rows, used

    best_rows_overall = None
    best_cost_overall = float('inf')
    
    start_time_global = time.time()
    max_total_time = 110
    
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() - start_time_global > max_total_time - 2:
            break
            
        rows, used = greedy_init_randomized()
        
        current_rows = [r[:] for r in rows]
        current_used = set(used)
        current_cost = max(max(r) for r in current_rows)
        
        best_rows = [r[:] for r in current_rows]
        best_cost = current_cost
        
        T = 5.0
        T_min = 0.01
        cooling = 0.9998
        
        start_time = time.time()
        remaining_restarts = n_restarts - restart
        max_time = (max_total_time - (time.time() - start_time_global)) / remaining_restarts
        
        while time.time() - start_time < max_time and T > T_min:
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            p_idx = random.randint(1, row_len - 1)
            
            low = row[p_idx - 1] + 1
            high = row[p_idx + 1] - 1 if p_idx < row_len - 1 else current_cost + 10
            
            if high < low:
                T *= cooling
                continue
                
            new_val = random.randint(low, high)
            
            if new_val == row[p_idx]:
                T *= cooling
                continue
                
            old_diffs = remove_point_diffs(row, p_idx)
            
            new_diffs = set()
            for j, v in enumerate(row):
                if j != p_idx:
                    new_diffs.add(abs(new_val - v))
                    
            if len(new_diffs) != row_len - 1:
                T *= cooling
                continue
                
            valid = True
            for d in new_diffs:
                if d in current_used and d not in old_diffs:
                    valid = False
                    break
                    
            if not valid:
                T *= cooling
                continue
                
            old_val = row[p_idx]
            row[p_idx] = new_val
            
            current_used -= old_diffs
            current_used |= new_diffs
            
            new_cost = max(max(r) for r in current_rows)
            
            delta_cost = new_cost - current_cost
            if delta_cost <= 0:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [r[:] for r in current_rows]
            else:
                if random.random() < math.exp(-delta_cost / T):
                    current_cost = new_cost
                else:
                    row[p_idx] = old_val
                    current_used -= new_diffs
                    current_used |= old_diffs
            
            T *= cooling
            
        if best_cost < best_cost_overall:
            best_cost_overall = best_cost
            best_rows_overall = [r[:] for r in best_rows]
            
    return best_rows_overall
