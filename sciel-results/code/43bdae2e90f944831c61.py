import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = [False] * 500  # Safe upper bound for differences
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if used[d]:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                row.append(x)
                for d in new_diffs:
                    used[d] = True
        rows.append(row)
    
    return rows

def _row_diffs_list(row):
    """Compute the list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _compute_all_used(rows, max_val):
    """Compute the boolean array of used differences."""
    all_used = [False] * (max_val + 50)
    for row in rows:
        diffs = _row_diffs_list(row)
        for d in diffs:
            if d < len(all_used):
                all_used[d] = True
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
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # 2. Simulated Annealing
    current_rows = [r[:] for r in rows]
    current_val = best_val
    
    # Use a boolean array for used diffs for speed
    max_possible = best_val + 100
    all_used = _compute_all_used(current_rows, max_possible)
    
    # Precompute row diffs as lists for fast diff calculation
    row_diffs = [_row_diffs_list(r) for r in current_rows]
    
    T = 500.0
    T_min = 0.5
    cool = 0.99998
    
    no_improve = 0
    max_no_improve = 50000
    steps = 0
    
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
                
            # Check if new_val is within current used array bounds
            # If we go too high, we might need to extend all_used, but for now assume it's ok or reject if too high
            # We want to minimize, so high values are less likely to be accepted unless T is high
            if new_val > len(all_used) - 50:
                continue

            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Calculate new diffs for this row
            new_diffs = _row_diffs_list(cand_row)
            
            # Check internal validity (no duplicate diffs in row)
            # Since the row is strictly increasing, duplicates can only happen if the set of differences has duplicates
            # _row_diffs_list returns a list. We check for uniqueness.
            if len(set(new_diffs)) != len(new_diffs):
                continue
                
            old_diffs = row_diffs[ri]
            
            # Check for conflicts with other rows
            conflict = False
            for d in new_diffs:
                if d >= len(all_used):
                    # If diff is out of bounds, it's new, so no conflict with existing used
                    # But we should probably extend the array or just treat it as unused
                    # For simplicity, if it's out of bounds, it's definitely not in all_used
                    continue
                if all_used[d]:
                    # Check if this diff was already in the old row
                    # If it was in the old row, it's fine (it's being replaced)
                    # But we need to be careful: if the diff appears multiple times in old_diffs? No, row is valid.
                    # So if d is in all_used, it must have come from this row or another.
                    # If it came from this row, it will be removed.
                    # We need to know if d is in old_diffs.
                    # Doing `d in old_diffs` is O(k^2).
                    # Better: maintain a set or count for old_diffs?
                    # For now, let's just check if d is in old_diffs.
                    if d not in old_diffs:
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
                for d in old_diffs:
                    if d < len(all_used):
                        all_used[d] = False
                # Add new diffs to all_used
                for d in new_diffs:
                    if d >= len(all_used):
                        # Extend if necessary
                        all_used.extend([False] * (d - len(all_used) + 10))
                    all_used[d] = True
                    
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
            all_used = _compute_all_used(current_rows, current_val + 100)
            row_diffs = [_row_diffs_list(r) for r in current_rows]
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
        all_used = _compute_all_used(current_rows, target_val + 50)
        row_diffs = [_row_diffs_list(r) for r in current_rows]
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
            
            new_diffs = _row_diffs_list(cand_row)
            if len(set(new_diffs)) != len(new_diffs):
                continue
                
            old_diffs = row_diffs[ri]
            
            conflict = False
            for d in new_diffs:
                if d >= len(all_used):
                    continue
                if all_used[d] and d not in old_diffs:
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
                for d in old_diffs:
                    if d < len(all_used):
                        all_used[d] = False
                for d in new_diffs:
                    if d >= len(all_used):
                        all_used.extend([False] * (d - len(all_used) + 10))
                    all_used[d] = True
                    
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
                break # Restart inner loop logic? No, just continue SA
            
            no_improve += 1
            T = max(T_min, T * cool)
            
            if no_improve > 1000:
                break
                
        # If we found a better solution, update best_val and continue to next target
        # The loop structure above updates best_val if improved.
        # We should break if we reached the target or below
        if best_val <= target_val:
            continue # Try to go even lower
        else:
            # If we didn't improve, we can try the next lower target
            pass

    return best_rows

# EVOLVE-BLOCK-END
