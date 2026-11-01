import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, gap_factor=10, random_seed=None):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    If random_seed is provided, adds small random perturbations for diversity.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Occasionally try a slightly larger gap for diversity
            if random_seed is not None and random.random() < 0.15:
                x += random.randint(1, 3)
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
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

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _compute_new_diffs(cand_row, old_val, pi):
    """
    Compute new diffs for a row after changing position pi from old_val to cand_row[pi].
    Returns (new_diffs_set, is_valid_internal).
    Optimized: only compute diffs involving position pi, reuse others.
    """
    n = len(cand_row)
    new_val = cand_row[pi]
    
    # Build set of all diffs efficiently
    new_diffs = set()
    
    # Diffs not involving pi (unchanged from before, but we need full set for conflict check)
    # Actually, we need the full new set. Let's compute it efficiently.
    for i in range(n):
        for j in range(i + 1, n):
            d = cand_row[j] - cand_row[i]
            if d in new_diffs:
                return None, False
            new_diffs.add(d)
    
    return new_diffs, True

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_rows = None
    best_val = float('inf')
    
    # Multi-start: try several different initializations
    num_starts = 3
    time_per_start = (deadline - start_time) / num_starts
    
    for start_i in range(num_starts):
        start_deadline = start_time + (start_i + 1) * time_per_start
        
        # 1. Initialization with diversity
        if start_i == 0:
            rows = _greedy_init(n_rows, k, random_seed=None)
        else:
            rows = _greedy_init(n_rows, k, random_seed=start_i)
        
        # Track used differences for fast checking
        all_used = _compute_all_used(rows)
        
        cur_best_rows = [r[:] for r in rows]
        cur_best_val = _global_max(rows)
        
        # Maintain row maxes for efficient global max computation
        row_maxes = [max(r) for r in rows]
        
        # 2. Simulated Annealing
        current_rows = [r[:] for r in rows]
        current_val = cur_best_val
        T = 300.0
        T_min = 0.3
        cool = 0.99997
        
        no_improve = 0
        max_no_improve = 15000
        steps = 0
        
        # Precompute row diffs for each row
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        
        while time.time() < start_deadline and no_improve < max_no_improve:
            steps += 1
            
            # Pick a row and position to modify - bias toward rows with high max
            if random.random() < 0.6:
                # Pick row with highest max
                ri = row_maxes.index(max(row_maxes))
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)  # 1..k
            
            old_val = current_rows[ri][pi]
            
            # Generate candidates with bias toward reducing max
            candidates = []
            
            # Decreasing candidates - try more of these, prioritize larger decreases
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(80, old_val - min_allowed)
            # Try larger decreases first for bigger improvements
            for d in range(max_decrease, 0, -1):
                new_v = old_val - d
                candidates.append(new_v)
                
            # Some small increases for exploration
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
            max_increase = min(15, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                new_v = old_val + d
                candidates.append(new_v)
                
            # Random jump (large perturbation) - less frequent
            if random.random() < 0.1:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 25
                if low <= high:
                    candidates.append(random.randint(low, high))
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            random.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                # Basic validity check for this position
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                    
                # Build candidate row
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
                # Calculate new diffs for this row
                new_diffs, valid_internal = _compute_new_diffs(cand_row, old_val, pi)
                
                if not valid_internal:
                    continue
                    
                # Get old diffs of row ri
                old_diffs = row_diffs[ri]
                
                # Check for conflicts: new_diffs should not intersect with (all_used - old_diffs)
                conflict = False
                for d in new_diffs:
                    if d in all_used and d not in old_diffs:
                        conflict = True
                        break
                
                if conflict:
                    continue
                    
                # Move is valid - compute new global max efficiently
                cand_row_max = max(cand_row)
                
                # Update row_maxes temporarily
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max  # restore
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.2:  # Allow some neutral moves to shuffle
                        accept = True
                else:
                    # Accept worse moves with probability
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
                    
                    if current_val < cur_best_val:
                        cur_best_val = current_val
                        cur_best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                        
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
                
            T = max(T_min, T * cool)
            
            # Periodic restart if stuck
            if no_improve > 8000:
                current_rows = [r[:] for r in cur_best_rows]
                current_val = cur_best_val
                all_used = _compute_all_used(current_rows)
                row_diffs = [_row_diffs_set(r) for r in current_rows]
                row_maxes = [max(r) for r in current_rows]
                T = 300.0
                no_improve = 0
        
        # Update global best
        if cur_best_val < best_val:
            best_val = cur_best_val
            best_rows = [r[:] for r in cur_best_rows]
    
    # 3. Final polishing phase with tighter tolerance
    if best_rows is not None:
        polish_deadline = min(deadline, start_time + 113.0)
        T = 0.5
        T_min = 0.05
        cool = 0.99999
        no_improve = 0
        max_no_improve_polish = 10000
        
        # Reset state to best
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used = _compute_all_used(current_rows)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]
        
        while time.time() < polish_deadline and no_improve < max_no_improve_polish:
            # Pick a row and position - bias toward max row
            if random.random() < 0.7:
                ri = row_maxes.index(max(row_maxes))
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            # Only try decreasing moves in polish phase
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(40, old_val - min_allowed)
            
            candidates = []
            for d in range(max_decrease, 0, -1):
                candidates.append(old_val - d)
            
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
                
                new_diffs, valid_internal = _compute_new_diffs(cand_row, old_val, pi)
                
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
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.1:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.02))
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
                        
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
                
            T = max(T_min, T * cool)
    
    return best_rows

# EVOLVE-BLOCK-END
