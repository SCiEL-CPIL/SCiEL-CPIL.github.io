import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115  # 5s buffer

    # 1. Initialize with a valid solution using a simple greedy
    # This ensures we start in the feasible region.
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            # Find next x that doesn't conflict
            # Simple greedy: pick the smallest valid x.
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
                x += 1
        rows.append(row)
    
    current_max = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        # Determine move
        # 50% chance to target the current global max
        if random.random() < 0.5:
            # Find a row and index with current_max
            candidates = []
            for r_idx, r in enumerate(rows):
                for i_idx, val in enumerate(r):
                    if val == current_max and i_idx > 0: # Don't move 0
                        candidates.append((r_idx, i_idx))
            
            if not candidates:
                continue
            
            r_idx, i_idx = random.choice(candidates)
            old_val = rows[r_idx][i_idx]
            
            # Try to reduce old_val
            lower = rows[r_idx][i_idx - 1] + 1
            upper = rows[r_idx][i_idx + 1] - 1 if i_idx < k else old_val - 1
            
            if upper < lower:
                continue
            
            # Sample new value, biased towards lower end
            new_val = random.randint(lower, upper)
        else:
            # Random move
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, k) # 0 is fixed
            
            old_val = rows[r_idx][i_idx]
            lower = rows[r_idx][i_idx - 1] + 1
            # Allow some expansion to escape local minima, but limit it
            upper = rows[r_idx][i_idx + 1] - 1 if i_idx < k else old_val + 10
            
            upper_bound = min(upper, current_max + 10)
            if upper_bound < lower:
                continue
            
            new_val = random.randint(lower, upper_bound)
        
        if new_val == old_val:
            continue
            
        # Check validity of moving rows[r_idx][i_idx] from old_val to new_val
        row = rows[r_idx]
        
        # 1. Identify old differences involving old_val
        old_diffs_to_remove = set()
        for j in range(i_idx):
            old_diffs_to_remove.add(old_val - row[j])
        for j in range(i_idx + 1, len(row)):
            old_diffs_to_remove.add(row[j] - old_val)
        
        # 2. Identify new differences involving new_val
        new_diffs_to_add = set()
        for j in range(i_idx):
            new_diffs_to_add.add(new_val - row[j])
        for j in range(i_idx + 1, len(row)):
            new_diffs_to_add.add(row[j] - new_val)
        
        # 3. Check conflict
        # Conflict if any d in new_diffs_to_add is in used AND d is not in old_diffs_to_remove
        conflict = False
        for d in new_diffs_to_add:
            if d in used and d not in old_diffs_to_remove:
                conflict = True
                break
        
        if conflict:
            continue
            
        # Valid move. Apply it.
        used -= old_diffs_to_remove
        used |= new_diffs_to_add
        
        # Update row
        row[i_idx] = new_val
        
        new_max = max(max(r) for r in rows)
        delta_cost = new_max - current_max
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            # Accept
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Reject: Revert
            row[i_idx] = old_val
            used -= new_diffs_to_add
            used |= old_diffs_to_remove
            
        T *= cooling

    return best_rows
