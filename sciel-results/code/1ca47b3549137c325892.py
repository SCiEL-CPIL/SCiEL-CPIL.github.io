import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, gap_factor=10):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
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

def _diffs_to_bytes(diffs, max_val):
    """Convert a set of differences to a bytes object for fast conflict checking."""
    arr = bytearray(max_val + 1)
    for d in diffs:
        if 0 < d <= max_val:
            arr[d] = 1
    return bytes(arr)

def _check_conflict_fast(new_diffs, old_diffs, all_used_bytes, max_val):
    """Check if new_diffs conflicts with all_used (excluding old_diffs)."""
    for d in new_diffs:
        if d in old_diffs:
            continue
        if d > max_val:
            continue
        if all_used_bytes[d]:
            return True
    return False

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _check_move_validity(cand_row, old_diffs, all_used):
    """
    Check if a candidate row is valid: no internal duplicate diffs,
    and no conflict with other rows' diffs.
    Returns (new_diffs, is_valid)
    """
    n = len(cand_row)
    new_diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = cand_row[j] - cand_row[i]
            if d in new_diffs:
                return None, False
            new_diffs.add(d)
    
    # Check for conflicts with other rows
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return None, False
    
    return new_diffs, True

def _compute_row_diffs_fast(cand_row):
    """Compute row diffs with early exit on internal collision. Returns (diffs, valid)."""
    n = len(cand_row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = cand_row[j] - cand_row[i]
            if d in diffs:
                return None, False
            diffs.add(d)
    return diffs, True

def _deterministic_polish(rows, all_used, row_diffs, row_maxes, time_limit=15.0):
    """
    Deterministic local search: try all single-position moves (decrease and increase).
    Repeatedly apply the best improving move until no improvement or time limit.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_val = max(row_maxes)
    best_val = current_val
    best_rows = [r[:] for r in current_rows]
    
    n_rows = len(current_rows)
    k = len(current_rows[0]) - 1
    
    # Track the global max entry for bounding search space
    global_max_entry = current_val
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        best_delta = 0
        best_ri = -1
        best_pi = -1
        best_new_val = -1
        best_new_diffs = None
        best_old_diffs = None
        
        for ri in range(n_rows):
            for pi in range(1, k + 1):
                old_val = current_rows[ri][pi]
                min_allowed = current_rows[ri][pi - 1] + 1
                
                # Try decreasing moves
                max_decrease = old_val - 1
                for new_val in range(min_allowed, max_decrease + 1):
                    if new_val >= old_val:
                        break
                    if pi < k and new_val >= current_rows[ri][pi + 1]:
                        continue
                    
                    cand_row = current_rows[ri][:]
                    cand_row[pi] = new_val
                    
                    new_diffs, valid = _compute_row_diffs_fast(cand_row)
                    if not valid:
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
                    if delta < best_delta:
                        best_delta = delta
                        best_ri = ri
                        best_pi = pi
                        best_new_val = new_val
                        best_new_diffs = new_diffs
                        best_old_diffs = old_diffs
        
        # Also try small increases (to enable subsequent decreases)
        for ri in range(n_rows):
            for pi in range(1, k + 1):
                old_val = current_rows[ri][pi]
                max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else global_max_entry + 5
                
                # Try a few increasing moves
                for new_val in range(old_val + 1, min(old_val + 10, max_allowed + 1)):
                    if pi < k and new_val >= current_rows[ri][pi + 1]:
                        break
                    
                    cand_row = current_rows[ri][:]
                    cand_row[pi] = new_val
                    
                    new_diffs, valid = _compute_row_diffs_fast(cand_row)
                    if not valid:
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
                    # Only accept increases if they could lead to a better decrease
                    if delta <= 2 and delta < best_delta:
                        best_delta = delta
                        best_ri = ri
                        best_pi = pi
                        best_new_val = new_val
                        best_new_diffs = new_diffs
                        best_old_diffs = old_diffs
        
        if best_ri >= 0 and best_delta < 0:
            # Apply the best move
            old_diffs = best_old_diffs
            all_used -= old_diffs
            all_used |= best_new_diffs
            current_rows[best_ri][best_pi] = best_new_val
            row_diffs[best_ri] = best_new_diffs
            row_maxes[best_ri] = max(current_rows[best_ri])
            current_val = max(row_maxes)
            global_max_entry = max(global_max_entry, current_val)
            
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
            improved = True
        elif best_ri >= 0 and best_delta >= 0:
            # Accept small increases to escape local minima
            old_diffs = best_old_diffs
            all_used -= old_diffs
            all_used |= best_new_diffs
            current_rows[best_ri][best_pi] = best_new_val
            row_diffs[best_ri] = best_new_diffs
            row_maxes[best_ri] = max(current_rows[best_ri])
            current_val = max(row_maxes)
            global_max_entry = max(global_max_entry, current_val)
            improved = True
    
    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    
    # Track used differences for fast checking
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # 2. Simulated Annealing with better schedule
    current_rows = [r[:] for r in rows]
    current_val = best_val
    
    # Better initial temperature based on problem size
    T = max(50.0, best_val * 0.5)
    T_min = 0.1
    # Adaptive cooling: slower early, faster later
    cool = 0.9999
    
    no_improve = 0
    max_no_improve = 50000
    steps = 0
    
    # Precompute row diffs for each row
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
    # Track restarts
    restart_count = 0
    max_restarts = 5
    
    while time.time() < deadline:
        steps += 1
        
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)  # 1..k
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates with bias toward reducing max
        candidates = []
        
        # Decreasing candidates - try fewer but targeted
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            new_v = old_val - d
            candidates.append(new_v)
            
        # Increasing candidates (for exploration) - fewer
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
        max_increase = min(5, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            new_v = old_val + d
            candidates.append(new_v)
            
        # Random jump (large perturbation)
        if random.random() < 0.05:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
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
            
            # Calculate new diffs for this row efficiently
            new_diffs, valid_internal = _compute_row_diffs_fast(cand_row)
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
                if random.random() < 0.3:  # Allow more neutral moves to shuffle
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
        if no_improve > 20000 and restart_count < max_restarts:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = max(50.0, best_val * 0.4)
            no_improve = 0
            restart_count += 1
    
    # 3. Final polishing phase with tighter tolerance
    polish_deadline = min(deadline, start_time + 140.0)
    T = 0.5
    T_min = 0.05
    cool = 0.99995
    no_improve = 0
    max_no_improve_polish = 15000
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        # Pick a row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Try both decreasing and small increasing moves in polish phase
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(15, old_val - min_allowed)
        
        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        # Small increases to escape local minima
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 5
        max_increase = min(3, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
        
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
            
            new_diffs, valid_internal = _compute_row_diffs_fast(cand_row)
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
                if random.random() < 0.2:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.05))
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
    
    # 4. Deterministic final polish: exact local optimization
    # Ensure we have at least 5 seconds left for the polish and return
    if time.time() < start_time + 105.0:
        remaining_time = min(5.0, start_time + 105.0 - time.time())
        if remaining_time > 0:
            best_rows, best_val = _deterministic_polish(
                best_rows, 
                _compute_all_used(best_rows),
                [_row_diffs_set(r) for r in best_rows],
                [max(r) for r in best_rows],
                time_limit=remaining_time
            )
    
    return best_rows

# EVOLVE-BLOCK-END
