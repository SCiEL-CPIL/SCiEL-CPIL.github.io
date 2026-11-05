import time
import random
import sys
from bisect import bisect_left, insort

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = 140.0  # Safe margin before 150s kill
    
    # Total number of differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Helper to compute all differences in a row
    def row_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    # Helper to check if a row is valid given used_diffs
    def is_valid_row(row, used_diffs):
        for d in row_diffs(row):
            if d in used_diffs:
                return False
        return True
    
    # Greedy construction with a given scope
    def greedy_construct(scope):
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            current_row = [0]
            for _ in range(k):
                last_val = current_row[-1]
                found = False
                for x in range(last_val + 1, scope + 1):
                    new_diffs = {x - a for a in current_row}
                    if not (new_diffs & used_diffs):
                        current_row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                if not found:
                    return None
            rows.append(current_row)
        return rows
    
    # Try to find an initial solution
    initial_rows = None
    scope_try = 120
    while initial_rows is None and scope_try <= 300:
        initial_rows = greedy_construct(scope_try)
        if initial_rows is None:
            scope_try += 5
    
    if initial_rows is None:
        # Fallback: extremely loose greedy
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            current_row = [0]
            for _ in range(k):
                last_val = current_row[-1]
                x = last_val + 1
                while True:
                    new_diffs = {x - a for a in current_row}
                    if not (new_diffs & used_diffs):
                        current_row.append(x)
                        used_diffs.update(new_diffs)
                        break
                    x += 1
            rows.append(current_row)
        initial_rows = rows
    
    # State representation
    current_rows = [row[:] for row in initial_rows]
    
    # Compute current scope (largest entry)
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    current_scope = get_scope(current_rows)
    best_scope = current_scope
    best_rows = [row[:] for row in current_rows]
    
    # Simulated Annealing
    T = 20.0
    cooling = 0.99995
    min_T = 0.01
    
    # Helper to get diffs for a specific index in a row
    def get_diffs_for_index(row, idx):
        diffs = []
        for i in range(idx):
            diffs.append(row[idx] - row[i])
        for j in range(idx + 1, len(row)):
            diffs.append(row[j] - row[idx])
        return diffs
    
    # Main SA loop
    steps = 0
    last_improve_step = 0
    no_improve_count = 0
    
    # Precompute used_diffs for current state
    def compute_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used
    
    used_diffs = compute_used_diffs(current_rows)
    
    while time.time() - start_time < deadline:
        # Generate a random move
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k)  # Indices 1..k
        
        row = current_rows[r_idx]
        old_val = row[j_idx]
        
        # Determine valid range for new value
        lower_bound = row[j_idx - 1] + 1
        if j_idx < k:
            upper_bound = row[j_idx + 1] - 1
        else:
            # Allow some expansion, but not too much
            upper_bound = current_scope + 15
            
        if lower_bound > upper_bound:
            continue
            
        new_val = random.randint(lower_bound, upper_bound)
        
        if new_val == old_val:
            continue
            
        # Calculate delta
        old_diffs = get_diffs_for_index(row, j_idx)
        
        # Temporarily set new val
        row[j_idx] = new_val
        new_diffs = get_diffs_for_index(row, j_idx)
        
        # Check validity: all new_diffs must not be in used_diffs (except those that were in old_diffs)
        # Remove old_diffs from used_diffs
        for d in old_diffs:
            used_diffs.discard(d)
        
        # Check if new_diffs conflict
        valid = True
        for d in new_diffs:
            if d in used_diffs:
                valid = False
                break
        
        if valid:
            # Add new_diffs to used_diffs
            for d in new_diffs:
                used_diffs.add(d)
            
            new_scope = get_scope(current_rows)
            delta_scope = new_scope - current_scope
            
            # Acceptance
            accept = False
            if delta_scope < 0:
                accept = True
            else:
                if T > 0:
                    if delta_scope / T < 700:
                        accept = random.random() < (2.718281828 ** (-delta_scope / T))
                    else:
                        accept = False
        
        if accept:
            current_scope = get_scope(current_rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [row[:] for row in current_rows]
                last_improve_step = steps
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert changes
            # Remove new_diffs
            for d in new_diffs:
                used_diffs.discard(d)
            
            # Restore old val
            row[j_idx] = old_val
            
            # Restore old_diffs
            for d in old_diffs:
                used_diffs.add(d)
            
            no_improve_count += 1
        
        steps += 1
        
        # Cooling
        if T > min_T:
            T *= cooling
        
        # Restart if stuck
        if no_improve_count > 5000:
            T = 20.0
            no_improve_count = 0
            # Randomize a bit to escape local minima
            for _ in range(3):
                r_idx = random.randint(0, n_rows - 1)
                j_idx = random.randint(1, k)
                row = current_rows[r_idx]
                old_val = row[j_idx]
                
                lower_bound = row[j_idx - 1] + 1
                if j_idx < k:
                    upper_bound = row[j_idx + 1] - 1
                else:
                    upper_bound = current_scope + 15
                
                if lower_bound > upper_bound:
                    continue
                
                new_val = random.randint(lower_bound, upper_bound)
                if new_val == old_val:
                    continue
                
                old_diffs = get_diffs_for_index(row, j_idx)
                row[j_idx] = new_val
                new_diffs = get_diffs_for_index(row, j_idx)
                
                for d in old_diffs:
                    used_diffs.discard(d)
                
                valid = True
                for d in new_diffs:
                    if d in used_diffs:
                        valid = False
                        break
                
                if valid:
                    for d in new_diffs:
                        used_diffs.add(d)
                    current_scope = get_scope(current_rows)
                else:
                    # Revert
                    for d in new_diffs:
                        used_diffs.discard(d)
                    row[j_idx] = old_val
                    for d in old_diffs:
                        used_diffs.add(d)

    return best_rows
