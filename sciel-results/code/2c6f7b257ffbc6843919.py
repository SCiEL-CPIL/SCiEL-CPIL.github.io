import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = [False] * 800  # Safe upper bound for differences
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d >= len(used):
                    used.extend([False] * (d - len(used) + 20))
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

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used_set(rows):
    """Compute the union of all differences across all rows as a set."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _compute_all_used_bool(rows, max_val):
    """Compute the boolean array of used differences."""
    all_used = [False] * (max_val + 100)
    for row in rows:
        diffs = _row_diffs_list(row)
        for d in diffs:
            if d < len(all_used):
                all_used[d] = True
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _incremental_diffs(cand_row, pi, old_val):
    """
    Compute the diffs for a row after changing position pi.
    Returns (new_diffs_set, is_valid_internal).
    """
    n = len(cand_row)
    new_diffs = set()
    
    # Differences involving position pi
    for i in range(n):
        if i == pi:
            continue
        d = abs(cand_row[pi] - cand_row[i])
        if d in new_diffs:
            return None, False
        new_diffs.add(d)
        
    # Differences not involving pi (these are unchanged, but we need the full set for conflict checking)
    # Actually, for the "new" diffs that need to be checked against "old" diffs, 
    # we only need the ones that changed. But for the final state, we need the full set.
    # Let's just compute the full set for simplicity in the candidate check, 
    # but we can optimize by only checking the new ones against the global set minus the old row's set.
    
    # Re-computing full set is O(k^2). 
    # A faster way: 
    # old_diffs is known.
    # removed_diffs = {old_val - a for a in row if a != old_val}
    # added_diffs = {new_val - a for a in row if a != new_val}
    # new_diffs = (old_diffs - removed_diffs) | added_diffs
    
    # Let's stick to the full re-calculation for correctness and simplicity, 
    # as k is small (6+1=7 elements, 21 diffs).
    
    # Wait, the previous function _compute_new_diffs in inspiration 1 did full re-calc.
    # Let's do full re-calc to ensure internal validity.
    for i in range(n):
        for j in range(i + 1, n):
            d = cand_row[j] - cand_row[i]
            if d in new_diffs:
                return None, False
            new_diffs.add(d)
            
    return new_diffs, True

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
    
    # Use a set for used diffs for O(1) lookups and easy updates
    all_used = _compute_all_used_set(current_rows)
    
    # Precompute row diffs as sets for fast diff calculation and removal
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
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
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Calculate new diffs for this row
            new_diffs, valid_internal = _incremental_diffs(cand_row, pi, old_val)
            
            if not valid_internal:
                continue
                
            old_diffs = row_diffs[ri]
            
            # Check for conflicts with other rows
            # Conflict if any new diff is in all_used but NOT in old_diffs
            # We can iterate over new_diffs and check if it's in all_used and not in old_diffs
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
                all_used -= old_diffs
                # Add new diffs to all_used
                all_used |= new_diffs
                
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
            
            new_diffs, valid_internal = _incremental_diffs(cand_row, pi, old_val)
            if not valid_internal:
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
                all_used -= old_diffs
                all_used |= new_diffs
                
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
            else:
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
