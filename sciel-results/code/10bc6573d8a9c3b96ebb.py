import random
import math
import time

def construct(n_rows, k):
    # 1. Initial Solution: Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    def calc_cost(rows):
        used = set()
        conflicts = 0
        max_val = 0
        for r in rows:
            max_val = max(max_val, max(r))
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
        return max_val + conflicts * 1000, max_val, conflicts

    current_cost, cur_m, cur_conf = calc_cost(rows)
    
    T = 100.0
    cooling = 0.999
    min_T = 0.1
    start_time = time.time()
    time_limit = 110
    
    while T > min_T and (time.time() - start_time) < time_limit:
        r_idx = random.randint(0, n_rows - 1)
        # Choose an index to modify. 
        # Modifying index 0 (value 0) is not allowed.
        # So we pick a value from the row (excluding 0) and replace it.
        # To ensure we can pick any value, we pick an index j from 1 to k.
        j_idx = random.randint(1, k)
        
        move_type = random.random()
        old_val = rows[r_idx][j_idx]
        new_val = old_val
        
        if move_type < 0.5:
            # Small change
            delta = random.choice([-2, -1, 1, 2])
            new_val = old_val + delta
        elif move_type < 0.8:
            # Medium change
            delta = random.randint(-5, 5)
            new_val = old_val + delta
        else:
            # Large change / Jump
            # Try to place it near other values to fix collisions
            # Or just random in a range
            new_val = random.randint(1, current_max + 10)
            
        # Apply tentative change
        # The row is a sorted list. 
        # We remove old_val and insert new_val, then sort.
        # This effectively moves the element to its correct sorted position.
        
        row = rows[r_idx]
        
        # Check validity
        if new_val <= 0:
            continue
        if new_val in row:
            continue
            
        # Remove old
        try:
            row.remove(old_val)
        except ValueError:
            # Should not happen
            continue
            
        # Insert new
        row.append(new_val)
        row.sort()
        
        new_cost, new_m, new_conf = calc_cost(rows)
        
        if new_conf == 0 and new_m < best_max:
            best_max = new_m
            best_rows = [r[:] for r in rows]
            
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            # Keep change
        else:
            # Revert
            row.remove(new_val)
            row.append(old_val)
            row.sort()
            
        T *= cooling
        
    return best_rows
