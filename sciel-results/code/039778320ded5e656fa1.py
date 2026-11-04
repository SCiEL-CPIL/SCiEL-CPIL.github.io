import time
import random
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Greedy Initialization
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
                used.update(new_diffs)
        rows.append(row)
        
    best_rows = copy.deepcopy(rows)
    best_scope = max(max(r) for r in rows)
    
    # Precompute row diffs
    row_diffs = [set() for _ in range(n_rows)]
    maxes = [0] * n_rows
    for i, row in enumerate(rows):
        diffs = set()
        for j in range(len(row)):
            for l in range(j):
                diffs.add(row[j] - row[l])
        row_diffs[i] = diffs
        maxes[i] = max(row)
        
    # 2. Local Search / Simulated Annealing
    T = 50.0
    T_min = 0.01
    cooling = 0.9999
    no_improve_count = 0
    max_no_improve = 5000
    best_scope_ever = best_scope
    
    while time.time() < deadline and T > T_min:
        # Pick row: biased towards larger max
        weights = [m + 1 for m in maxes]
        r_idx = random.choices(range(n_rows), weights=weights, k=1)[0]
        
        # Pick position
        j = random.randint(1, k)
        
        # Generate candidate v
        current_val = rows[r_idx][j]
        lower = rows[r_idx][j-1] + 1
        upper = rows[r_idx][j+1] - 1 if j < k else current_val + 10
        
        candidates = []
        # Try shrinking
        for d in range(1, 8):
            v = current_val - d
            if v >= lower:
                candidates.append(v)
        # Try random in valid range
        if lower <= upper:
            candidates.append(random.randint(lower, upper))
            
        if not candidates:
            T *= cooling
            continue
            
        v = random.choice(candidates)
        
        # Save old state
        old_row = rows[r_idx][:]
        old_diffs = row_diffs[r_idx]
        old_max = maxes[r_idx]
        
        # Remove old diffs from global used
        used.difference_update(old_diffs)
        
        # Create new row
        new_row = old_row[:]
        new_row[j] = v
        
        # Check strictly increasing
        valid = True
        for idx in range(len(new_row)-1):
            if new_row[idx] >= new_row[idx+1]:
                valid = False
                break
                
        if not valid:
            used.update(old_diffs)
            T *= cooling
            continue
            
        # Calculate new diffs
        new_diffs = set()
        for idx in range(len(new_row)):
            for l in range(idx):
                new_diffs.add(new_row[idx] - new_row[l])
                
        # Check for internal duplicates
        if len(new_diffs) != 15:
            used.update(old_diffs)
            T *= cooling
            continue
            
        # Check conflicts with other rows
        if new_diffs & used:
            used.update(old_diffs)
            T *= cooling
            continue
            
        # Valid move
        # Update global state
        used.update(new_diffs)
        rows[r_idx] = new_row
        row_diffs[r_idx] = new_diffs
        
        new_max = max(new_row)
        maxes[r_idx] = new_max
        
        old_scope = max(maxes)
        new_scope = max(maxes)
        
        # Acceptance
        if new_scope < old_scope:
            accept = True
            no_improve_count = 0
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = copy.deepcopy(rows)
        else:
            delta = new_scope - old_scope
            if delta == 0:
                accept = random.random() < 0.5
            else:
                if delta > 1000:
                    accept = False
                else:
                    accept = random.random() < (2.718281828 ** (-delta / T))
                    
        if not accept:
            # Revert
            used.difference_update(new_diffs)
            rows[r_idx] = old_row
            row_diffs[r_idx] = old_diffs
            maxes[r_idx] = old_max
            no_improve_count += 1
        else:
            no_improve_count = 0
            
        T *= cooling
        
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            T = max(T, 10.0)

    return best_rows
