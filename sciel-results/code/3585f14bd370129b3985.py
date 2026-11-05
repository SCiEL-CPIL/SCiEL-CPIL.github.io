import time
import random
import math
import sys

def _row_diffs_list(row):
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _row_mask(row):
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))

def _greedy_construct(n_rows, num_elements, rng, max_scope_hint=200):
    """
    Greedy construction: build rows one by one, picking smallest feasible values.
    """
    rows = []
    used_mask = 0
    
    for _ in range(n_rows):
        row = [0]
        last_val = 0
        # Try to build a row with max value <= max_scope_hint
        # We use a simple greedy: pick smallest x > last_val such that diffs are new
        # and not in used_mask
        found = True
        while len(row) < num_elements:
            next_val = last_val + 1
            valid = False
            while next_val <= max_scope_hint:
                new_diffs = [next_val - prev for prev in row]
                # Check internal uniqueness
                if len(set(new_diffs)) != len(new_diffs):
                    next_val += 1
                    continue
                # Check global uniqueness
                mask_new = 0
                for d in new_diffs:
                    mask_new |= (1 << d)
                if (mask_new & used_mask) == 0:
                    valid = True
                    break
                next_val += 1
            
            if not valid:
                found = False
                break
            
            row.append(next_val)
            last_val = next_val
            # Update used_mask
            for d in [next_val - prev for prev in row[:-1]]:
                used_mask |= (1 << d)
        
        if found:
            rows.append(row)
        else:
            # If greedy fails, try a slightly looser bound or restart
            # For simplicity in initialization, just extend the bound
            # But this greedy is very restrictive.
            # Let's use a more robust backtracking for the first row if needed, 
            # but usually greedy works for initialization if scope is large enough.
            # To ensure we get a valid start, we might need to retry with higher scope.
            return None
            
    return rows

def _backtrack_row(prefix, used_mask, num_elements, max_val, rng, time_limit):
    """
    Backtrack to complete a row given a prefix and a forbidden difference mask.
    """
    if time.time() > time_limit:
        return None
    
    if len(prefix) == num_elements:
        return prefix[:]
    
    row = prefix
    last_val = row[-1]
    
    # Candidates: try smaller values first, but shuffle to escape local minima
    candidates = list(range(last_val + 1, max_val + 1))
    # Heuristic: prefer values that are not too close to last_val to spread diffs?
    # Or just randomize.
    rng.shuffle(candidates)
    
    for val in candidates:
        new_diffs = []
        valid = True
        for prev in row:
            d = val - prev
            if d <= 0:
                valid = False
                break
            if d in new_diffs:
                valid = False
                break
            new_diffs.append(d)
        
        if not valid:
            continue
        
        new_mask = 0
        for d in new_diffs:
            new_mask |= (1 << d)
        
        if new_mask & used_mask:
            continue
        
        result = _backtrack_row(row + [val], used_mask | new_mask, num_elements, max_val, rng, time_limit)
        if result is not None:
            return result
    
    return None

def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin
    
    num_elements = k + 1
    total_diffs_needed = n_rows * (num_elements * (num_elements - 1) // 2)
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Get a valid starting solution
    # Try greedy with increasing scope bounds
    for scope_bound in range(counting_bound, counting_bound + 100, 5):
        rows = _greedy_construct(n_rows, num_elements, rng, scope_bound)
        if rows and _is_valid(rows):
            best_rows = rows
            best_scope = _scope(rows)
            break
    
    # If greedy failed completely, use a pure backtracking construction for the whole set
    if best_rows is None:
        # This is a fallback, might be slow
        rows = []
        used_mask = 0
        for i in range(n_rows):
            # Backtrack a row given the current used_mask
            # We need to find a row that fits in a reasonable scope
            # Try scope from counting_bound upwards
            found = False
            for sb in range(counting_bound, counting_bound + 50):
                row = _backtrack_row([0], used_mask, num_elements, sb, rng, DEADLINE - 10)
                if row:
                    rows.append(row)
                    # Update used_mask
                    m = _row_mask(row)
                    used_mask |= m
                    found = True
                    break
            if not found:
                # Fail safe: return a simple valid set (even if large)
                # Just construct sequentially without optimization
                final_rows = []
                um = 0
                for _ in range(n_rows):
                    r = [0]
                    x = 0
                    while len(r) < num_elements:
                        x += 1
                        nd = {x - a for a in r}
                        if not (nd & um):
                            r.append(x)
                            um |= nd
                    final_rows.append(r)
                return final_rows
        best_rows = rows
        best_scope = _scope(rows)
    
    # Phase 2: Local Search / ILS
    # We want to minimize the scope.
    # State: list of rows, global mask.
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    # Precompute masks
    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
    
    # ILS Parameters
    T_start = 100.0
    T_end = 1.0
    cooling_rate = 0.999
    max_iterations = 100000
    
    T = T_start
    
    for it in range(max_iterations):
        if time.time() > DEADLINE - 5:
            break
        
        # 1. Perturb: Pick a random row and regenerate it with a slightly tighter scope
        # or pick a random row and try to compress it.
        
        # Strategy A: Compress a random row
        r_idx = rng.randint(0, n_rows - 1)
        old_row = current_rows[r_idx]
        old_mask = row_masks[r_idx]
        other_mask = global_mask & ~old_mask
        
        # Try to find a better row (smaller max) that is compatible with other_mask
        # Max value of old row is old_row[-1]
        # Try to find a row with max <= old_row[-1] - 1
        target_max = old_row[-1] - 1
        if target_max < num_elements - 1:
            target_max = num_elements - 1
            
        new_row = None
        # Use backtracking to find a row with max <= target_max
        # To speed up, limit the search depth/time per attempt
        # We can also try random restarts for the row construction
        for attempt in range(10):
            local_rng = random.Random(it * 100 + attempt)
            # Add some randomness to the candidate order in backtracking
            new_row = _backtrack_row([0], other_mask, num_elements, target_max, local_rng, time.time() + 0.1)
            if new_row:
                break
        
        if new_row:
            new_scope = _scope(current_rows) # Scope is max of all rows
            # Actually, we need to check if this new row reduces the global scope
            # Global scope is max(r[-1] for r in current_rows)
            # If new_row[-1] < current_scope, it might improve.
            # Even if not, it might allow other rows to be compressed later.
            
            new_mask = _row_mask(new_row)
            # Check validity (should be guaranteed by backtracking, but double check)
            if (new_mask & other_mask) == 0:
                # Calculate new global scope
                # If new_row is smaller, scope might drop if it was the max.
                # But we accept moves that reduce scope, and with prob exp(-delta/T) for SA.
                
                # Delta in objective function:
                # Objective: Minimize Scope.
                # Old Scope: current_scope
                # New Scope: max(new_row[-1], max(r[-1] for i, r in enumerate(current_rows) if i != r_idx))
                
                other_max = 0
                for i, r in enumerate(current_rows):
                    if i != r_idx:
                        if r[-1] > other_max:
                            other_max = r[-1]
                
                new_global_scope = max(new_row[-1], other_max)
                delta = new_global_scope - current_scope
                
                if delta < 0 or (delta == 0 and rng.random() < 0.1) or (rng.random() < math.exp(-delta / T)):
                    # Accept
                    current_rows[r_idx] = new_row
                    row_masks[r_idx] = new_mask
                    global_mask = (global_mask & ~old_mask) | new_mask
                    current_scope = new_global_scope
                    
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
        
        # Strategy B: Swap two rows with two other catalog rows?
        # We don't have a catalog. We can generate random rows.
        # Let's stick to Strategy A for now, it's robust.
        
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
