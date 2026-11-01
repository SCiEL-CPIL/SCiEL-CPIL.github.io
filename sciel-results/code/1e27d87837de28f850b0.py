import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Greedy initialization: each row takes the smallest next mark with no difference collision.
    Returns a list of rows.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in used:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                row.append(x)
                for d in new_diffs:
                    used.add(d)
        rows.append(row)
    return rows

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used_set(rows):
    """Compute the set of all used differences across all rows."""
    all_used = set()
    for row in rows:
        all_used.update(_row_diffs_set(row))
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    current_rows = [r[:] for r in rows]
    current_val = best_val
    
    # Precompute row diffs as sets for fast membership testing
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    all_used = _compute_all_used_set(current_rows)
    row_maxes = [max(r) for r in current_rows]
    
    # 2. Simulated Annealing with faster inner loop
    T = 500.0
    T_min = 0.5
    cool = 0.99998
    
    no_improve = 0
    max_no_improve = 50000
    steps = 0
    
    # Precompute the number of differences per row for quick checks
    # Each row has k+1 elements, so C(k+1, 2) differences
    num_diffs_per_row = (k + 1) * k // 2
    
    while time.time() < deadline and no_improve < max_no_improve:
        steps += 1
        
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)  # 1..k
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        candidates = []
        
        # Decreasing candidates
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(80, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        # Increasing candidates
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
        max_increase = min(30, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
        # Random jump
        if random.random() < 0.3:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            # Basic validity check
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Calculate new diffs for this row
            new_diffs = _row_diffs_set(cand_row)
            
            # Check internal validity (no duplicate diffs in row)
            # Since we're using a set, if len(new_diffs) != num_diffs_per_row, there were duplicates
            if len(new_diffs) != num_diffs_per_row:
                continue
                
            old_diffs = row_diffs[ri]
            
            # Check for conflicts with other rows
            # A diff is a conflict if it's in all_used but NOT in old_diffs
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Move is valid
            cand_row_max = max(cand_row)
            
            # Update row_maxes temporarily
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.3:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                # Remove old diffs from all_used
                all_used.difference_update(old_diffs)
                # Add new diffs to all_used
                all_used.update(new_diffs)
                
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        # Periodic restart if stuck
        if no_improve > 15000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used_set(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 500.0
            no_improve = 0
    
    # 3. Final Polishing: Try to reduce scope by 1, 2, ...
    # We try to find a solution with max_val <= best_val - 1, then best_val - 2, etc.
    # We use a short SA run with a hard cap on the max value.
    
    current_best = best_val
    # Try to improve up to 5 steps or until time runs out
    for target_val in range(current_best - 1, max(100, current_best - 10), -1):
        if time.time() > deadline - 5:
            break
            
        # Initialize with best_rows
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used = _compute_all_used_set(current_rows)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]
        
        T = 100.0
        T_min = 0.1
        cool = 0.9999
        no_improve = 0
        max_steps = 5000 # Short run
        
        for _ in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            
            # Only allow moves that keep max <= target_val
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else target_val
            
            if min_allowed > max_allowed:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            # Try a random valid value in range
            new_val = random.randint(min_allowed, max_allowed)
            
            if new_val == old_val:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            new_diffs = _row_diffs_set(cand_row)
            if len(new_diffs) != num_diffs_per_row:
                continue
                
            old_diffs = row_diffs[ri]
            
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            cand_row_max = max(cand_row)
            if cand_row_max > target_val:
                continue
                
            # Accept if it improves or is neutral (within cap)
            # Since we are capping, any valid move within cap is good
            # We prefer moves that reduce the max
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                # Allow slight increases if it helps escape local minima, but strictly below target_val
                if new_global_max <= target_val:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used.difference_update(old_diffs)
                all_used.update(new_diffs)
                
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            
            no_improve += 1
            T = max(T_min, T * cool)
            
            if no_improve > 1000:
                break
                
        # If we found a better solution, update best_val and continue to next target
        if best_val <= target_val:
            continue # Try to go even lower
        else:
            # If we didn't improve, we can try the next lower target
            pass

    return best_rows

# EVOLVE-BLOCK-END
