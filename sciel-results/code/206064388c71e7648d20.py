import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Greedy initialization using a boolean array for speed.
    """
    rows = []
    used = [False] * 600
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
                    used.extend([False] * (d - len(used) + 10))
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
    return set(_row_diffs_list(row))

def _compute_all_used_set(rows):
    """Compute the set of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _check_conflicts(new_diffs, old_diffs, all_used):
    """
    Check if new_diffs conflict with all_used, excluding those in old_diffs.
    Returns True if there is a conflict.
    """
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return True
    return False

def _compute_new_diffs_fast(row):
    """
    Compute the set of differences for a row efficiently.
    """
    diffs = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # State for ILS
    current_rows = [r[:] for r in rows]
    current_val = best_val
    row_maxes = [max(r) for r in current_rows]
    
    all_used = _compute_all_used_set(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
    # Parameters for ILS
    T = 300.0
    T_min = 0.5
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 50000
    perturbation_strength = 1.0
    
    # Main ILS loop
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        candidates = []
        
        # Decreasing candidates
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        # Increasing candidates
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
        # Random jump with perturbation
        if random.random() < 0.3:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
            if low <= high:
                # Wider jump for perturbation
                jump_range = int(10 * perturbation_strength)
                low_j = max(low, old_val - jump_range)
                high_j = min(high, old_val + jump_range)
                if low_j <= high_j:
                    candidates.append(random.randint(low_j, high_j))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Compute new diffs
            new_diffs_list = _row_diffs_list(cand_row)
            
            # Check internal validity
            if len(set(new_diffs_list)) != len(new_diffs_list):
                continue
                
            new_diffs = set(new_diffs_list)
            old_diffs = row_diffs[ri]
            
            # Check conflicts
            if _check_conflicts(new_diffs, old_diffs, all_used):
                continue
                
            # Move is valid
            cand_row_max = max(cand_row)
            
            # Calculate delta
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
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        # Restart if stuck with perturbation
        if no_improve > 8000:
            # Perturb the best solution
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used_set(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 300.0
            no_improve = 0
            
            # Apply perturbation: make several random valid moves
            for _ in range(3):
                ri = random.randrange(n_rows)
                pi = random.randrange(1, k + 1)
                old_val = current_rows[ri][pi]
                
                min_allowed = current_rows[ri][pi - 1] + 1
                max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 15
                if min_allowed > max_allowed:
                    continue
                
                new_val = random.randint(min_allowed, max_allowed)
                if new_val == old_val:
                    continue
                    
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
                new_diffs_list = _row_diffs_list(cand_row)
                if len(set(new_diffs_list)) != len(new_diffs_list):
                    continue
                    
                new_diffs = set(new_diffs_list)
                old_diffs = row_diffs[ri]
                
                if _check_conflicts(new_diffs, old_diffs, all_used):
                    continue
                    
                # Accept perturbation move regardless of cost
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = max(cand_row)
                current_val = max(row_maxes)
    
    # 2. Polishing Phase: Try to force lower max values
    polish_deadline = min(deadline, start_time + 147.0)
    
    current_best = best_val
    # Try to improve by 1, 2, 3...
    for target_offset in range(1, 6):
        target_val = current_best - target_offset
        if target_val < 100:  # Reasonable lower bound
            break
            
        if time.time() > polish_deadline - 2:
            break
            
        # Reset to best
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used = _compute_all_used_set(current_rows)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]
        
        T = 40.0
        T_min = 0.1
        cool = 0.9999
        no_improve = 0
        max_steps = 2000
        
        for _ in range(max_steps):
            if time.time() > polish_deadline - 2:
                break
                
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else target_val
            
            if min_allowed > max_allowed:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            new_val = random.randint(min_allowed, max_allowed)
            if new_val == old_val:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            new_diffs_list = _row_diffs_list(cand_row)
            if len(set(new_diffs_list)) != len(new_diffs_list):
                continue
                
            new_diffs = set(new_diffs_list)
            old_diffs = row_diffs[ri]
            
            if _check_conflicts(new_diffs, old_diffs, all_used):
                continue
                
            cand_row_max = max(cand_row)
            if cand_row_max > target_val:
                continue
                
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.4:
                    accept = True
            else:
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
                    # If we hit the target, we can break this target loop
                    if best_val <= target_val:
                        break
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            T = max(T_min, T * cool)
            
            if no_improve > 400:
                break
    
    return best_rows

# EVOLVE-BLOCK-END
