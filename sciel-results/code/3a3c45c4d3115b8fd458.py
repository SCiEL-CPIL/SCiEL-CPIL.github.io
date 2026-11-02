import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
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

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _random_valid_row(forbidden_diffs, max_limit, k, max_candidates=5, safety_bound=300):
    """
    Construct a random valid row of length k+1 starting at 0, strictly increasing,
    with all internal differences distinct and disjoint from forbidden_diffs,
    with max entry <= max_limit.
    """
    row = [0]
    used_in_row = set()
    last_val = 0
    
    for _ in range(k):
        candidates = []
        x = last_val + 1
        # Safety: don't search too far
        limit = min(max_limit, last_val + safety_bound) if max_limit is not None else last_val + safety_bound
        
        while x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                candidates.append(x)
            
            x += 1
            if len(candidates) >= max_candidates:
                break
        
        if not candidates:
            return None
        
        chosen_x = random.choice(candidates)
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_in_row |= new_diffs
        last_val = chosen_x
        
    return row

def _find_valid_decrease(rows, r_idx, pos, all_used, row_diffs, k):
    """
    Try to find a smaller valid value for rows[r_idx][pos].
    Returns (new_val, new_diffs) or (None, None) if no improvement found.
    """
    row = rows[r_idx]
    old_val = row[pos]
    n = k + 1
    
    # Compute diffs involving position pos
    diffs_involving = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving.add(old_val - row[i])
        else:
            diffs_involving.add(row[i] - old_val)
    
    forbidden_diffs = all_used - diffs_involving
    
    # Try to find smaller valid value
    lower_bound = row[pos-1] + 1
    upper_bound = old_val - 1
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos+1] - 1)
    
    if upper_bound < lower_bound:
        return None, None
    
    for new_val in range(lower_bound, upper_bound + 1):
        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            
            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if valid:
            return new_val, new_diffs
    
    return None, None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    
    # Track used differences for fast checking
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # Precompute row diffs for each row
    row_diffs = [_row_diffs_set(r) for r in rows]
    
    # Phase 1: Row-level reconstruction via randomized greedy
    # This allows escaping local minima where single-element moves are stuck
    current_rows = [r[:] for r in rows]
    current_val = best_val
    
    T = 50.0
    T_min = 1.0
    cool = 0.999
    
    no_improve = 0
    max_no_improve = 5000
    steps = 0
    
    while time.time() < deadline and no_improve < max_no_improve:
        steps += 1
        remaining_time = deadline - time.time()
        
        # Adapt strategy based on time remaining
        if remaining_time < 15.0:
            break
        
        # Pick a row to reconstruct
        ri = random.randrange(n_rows)
        
        # Compute forbidden diffs (all diffs from other rows)
        forbidden = set()
        for j in range(n_rows):
            if j != ri:
                forbidden |= row_diffs[j]
        
        # Try to find a better row
        # First try to find a row with max < current_val
        target_max = current_val - 1
        if target_max < k:
            target_max = current_rows[ri][-1] - 1
            
        new_row = _random_valid_row(forbidden, target_max, k, max_candidates=8)
        
        if new_row is None:
            target_max2 = current_val
            new_row = _random_valid_row(forbidden, target_max2, k, max_candidates=8)
            
        if new_row is None:
            target_max3 = current_val + 10
            new_row = _random_valid_row(forbidden, target_max3, k, max_candidates=5)
            
        if new_row is None:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        new_row_max = new_row[-1]
        
        other_maxes = [row_maxes[j] for j in range(n_rows) if j != ri]
        new_global_max = max(max(other_maxes) if other_maxes else 0, new_row_max)
        
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
            old_diffs = row_diffs[ri]
            new_diffs = _row_diffs_set(new_row)
            
            all_used -= old_diffs
            all_used |= new_diffs
            
            current_rows[ri] = new_row
            row_diffs[ri] = new_diffs
            row_maxes[ri] = new_row_max
            current_val = new_global_max
            
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        # Periodic restart if stuck
        if no_improve > max_no_improve // 2:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 50.0
            no_improve = 0
    
    # Phase 2: Element-level simulated annealing for fine-tuning
    polish_deadline = min(deadline, start_time + 108.0)
    T = 10.0
    T_min = 0.5
    cool = 0.99995
    no_improve = 0
    max_no_improve_polish = 20000
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        candidates = []
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
        max_increase = min(10, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
        if random.random() < 0.1:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 15
            if low <= high:
                candidates.append(random.randint(low, high))
        
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
            
            n = len(cand_row)
            new_diffs = set()
            valid_internal = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = cand_row[j] - cand_row[i]
                    if d in new_diffs:
                        valid_internal = False
                        break
                    new_diffs.add(d)
                if not valid_internal:
                    break
            
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
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        if no_improve > max_no_improve_polish // 2:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 10.0
            no_improve = 0
    
    # Phase 3: Intensive greedy descent pass - try to reduce each element to its minimum valid value
    final_deadline = min(deadline, start_time + 112.0)
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    # Multiple passes of greedy descent
    for pass_num in range(5):
        if time.time() > final_deadline:
            break
            
        improved = False
        # Try all positions in random order
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > final_deadline:
                break
                
            new_val, new_diffs = _find_valid_decrease(
                current_rows, r_idx, pos, all_used, row_diffs, k
            )
            
            if new_val is not None:
                # Apply the improvement
                old_val = current_rows[r_idx][pos]
                old_diffs = row_diffs[r_idx]
                
                # Compute diffs involving old position
                diffs_involving_old = set()
                n = k + 1
                for i in range(n):
                    if i == pos:
                        continue
                    if i < pos:
                        diffs_involving_old.add(old_val - current_rows[r_idx][i])
                    else:
                        diffs_involving_old.add(current_rows[r_idx][i] - old_val)
                
                all_used -= diffs_involving_old
                all_used |= new_diffs
                
                current_rows[r_idx][pos] = new_val
                row_diffs[r_idx] = new_diffs
                row_maxes[r_idx] = max(current_rows[r_idx])
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                
                improved = True
        
        if not improved:
            break
    
    # Phase 4: Final SA polish with low temperature
    T = 1.0
    T_min = 0.1
    cool = 0.99999
    no_improve = 0
    max_no_improve_final = 5000
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < final_deadline and no_improve < max_no_improve_final:
        max_val = current_val
        max_rows = [i for i in range(n_rows) if row_maxes[i] == max_val]
        
        if not max_rows:
            break
            
        ri = random.choice(max_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(15, old_val - min_allowed)
        
        candidates = []
        for d in range(1, max_decrease + 1):
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
            
            n = len(cand_row)
            new_diffs = set()
            valid_internal = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = cand_row[j] - cand_row[i]
                    if d in new_diffs:
                        valid_internal = False
                        break
                    new_diffs.add(d)
                if not valid_internal:
                    break
            
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
                if random.random() < 0.05:
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
    
    return best_rows

# EVOLVE-BLOCK-END
