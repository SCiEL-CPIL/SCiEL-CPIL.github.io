import time
import random
import math
import sys
from itertools import combinations

def _row_diffs_list(row):
    """Return list of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _row_diffs_set(row):
    """Return set of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))

def _compute_row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m

def _compute_all_diffs(rows):
    """Return (global_mask, row_masks) for all rows."""
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _compute_row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks

def _try_move_element(rows, row_idx, pos, new_val, global_mask, row_masks):
    """
    Try to move element at rows[row_idx][pos] to new_val.
    Returns (success, new_global_mask, new_row_mask) or (False, None, None).
    """
    row = rows[row_idx]
    n = len(row)
    
    if pos > 0 and new_val <= row[pos - 1]:
        return False, None, None
    if pos < n - 1 and new_val >= row[pos + 1]:
        return False, None, None
    
    old_val = row[pos]
    if new_val == old_val:
        return False, None, None
    
    row[pos] = new_val
    
    diffs_in_row = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs_in_row:
                row[pos] = old_val
                return False, None, None
            diffs_in_row.add(d)
    
    new_row_mask = 0
    for d in diffs_in_row:
        new_row_mask |= (1 << d)
    
    other_mask = global_mask & ~row_masks[row_idx]
    if new_row_mask & other_mask:
        row[pos] = old_val
        return False, None, None
    
    new_global_mask = (global_mask & ~row_masks[row_idx]) | new_row_mask
    row[pos] = old_val
    
    return True, new_global_mask, new_row_mask

def _greedy_construct(n_rows, num_elements, rng, target_scope):
    """
    Column-wise earliest-feasible construction with randomization.
    """
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    for pos in range(1, num_elements):
        # Order rows by tightness - rows with smaller last value first
        row_indices = list(range(n_rows))
        row_indices.sort(key=lambda r: rows[r][-1])
        
        # With some probability, shuffle to introduce diversity
        if rng.random() < 0.3:
            rng.shuffle(row_indices)
        
        success = True
        for r in row_indices:
            last_val = rows[r][-1]
            found = False
            
            # Try values in increasing order
            for val in range(last_val + 1, target_scope + 1):
                local_diffs = set()
                valid = True
                for i in range(len(rows[r])):
                    d = val - rows[r][i]
                    if d in used_diffs or d in local_diffs:
                        valid = False
                        break
                    local_diffs.add(d)
                
                if valid:
                    rows[r].append(val)
                    used_diffs.update(local_diffs)
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if not success:
            return None
    
    return rows

def _csp_backtrack(n_rows, num_elements, target_scope, rng, time_limit):
    """
    Randomized backtracking with degree heuristic.
    """
    start = time.time()
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    for pos in range(1, num_elements):
        if time.time() > time_limit:
            return None
        
        # Order rows by tightness (fewest feasible values first)
        row_order = list(range(n_rows))
        feasibilities = []
        for r in row_order:
            last_val = rows[r][-1]
            count = 0
            for val in range(last_val + 1, target_scope + 1):
                valid = True
                local_set = set()
                for i in range(len(rows[r])):
                    d = val - rows[r][i]
                    if d in used_diffs or d in local_set:
                        valid = False
                        break
                    local_set.add(d)
                if valid:
                    count += 1
            feasibilities.append((count, r))
        
        if any(f == 0 for f, r in feasibilities):
            return None
        
        feasibilities.sort()
        
        for f, r in feasibilities:
            if time.time() > time_limit:
                return None
            
            last_val = rows[r][-1]
            found = False
            
            candidates = list(range(last_val + 1, target_scope + 1))
            if rng.random() < 0.2:
                rng.shuffle(candidates)
            
            for val in candidates:
                local_diffs = set()
                valid = True
                for i in range(len(rows[r])):
                    d = val - rows[r][i]
                    if d in used_diffs or d in local_diffs:
                        valid = False
                        break
                    local_diffs.add(d)
                
                if valid:
                    rows[r].append(val)
                    used_diffs.update(local_diffs)
                    found = True
                    break
            
            if not found:
                return None
    
    return rows

def _simulated_annealing(rows, num_elements, rng, time_limit, initial_temp=30.0):
    """
    Simulated annealing to minimize scope with biased mutations towards smaller values.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    n_rows = len(current_rows)
    n_elem = num_elements
    
    temp = initial_temp
    cooling = 0.9998
    min_temp = 0.3
    
    while time.time() < time_limit and temp > min_temp:
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope + 5
        
        if low >= high:
            temp *= cooling
            continue
        
        # Bias towards smaller values (70% chance to search below current)
        if old_val > low + 1 and rng.random() < 0.7:
            search_high = min(old_val - 1, high)
            if low > search_high:
                new_val = rng.randint(low, high)
            else:
                new_val = rng.randint(low, search_high)
        else:
            new_val = rng.randint(low, high)
        
        if new_val == old_val:
            temp *= cooling
            continue
        
        success, new_global_mask, new_row_mask = _try_move_element(
            current_rows, row_idx, pos, new_val, global_mask, row_masks
        )
        
        if not success:
            temp *= cooling
            continue
        
        current_rows[row_idx][pos] = new_val
        old_row_mask = row_masks[row_idx]
        row_masks[row_idx] = new_row_mask
        global_mask = new_global_mask
        
        new_scope = _scope(current_rows)
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            current_rows[row_idx][pos] = old_val
            row_masks[row_idx] = old_row_mask
            global_mask = (global_mask & ~new_row_mask) | old_row_mask
        
        temp *= cooling
    
    return best_rows, best_scope

def _local_search_refine(rows, num_elements, rng, time_limit):
    """
    Local search: try to reduce scope by moving elements down.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    n_rows = len(current_rows)
    n_elem = num_elements
    
    stagnation = 0
    max_stagnation = 3000
    
    while time.time() < time_limit and stagnation < max_stagnation:
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope
        
        if low >= old_val:
            stagnation += 1
            continue
        
        new_val = rng.randint(low, old_val - 1)
        
        success, new_global_mask, new_row_mask = _try_move_element(
            current_rows, row_idx, pos, new_val, global_mask, row_masks
        )
        
        if success:
            current_rows[row_idx][pos] = new_val
            old_row_mask = row_masks[row_idx]
            row_masks[row_idx] = new_row_mask
            global_mask = new_global_mask
            
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
    
    return best_rows, best_scope

def _fallback_construct(n_rows, num_elements):
    """
    Simple greedy fallback that always produces a valid solution.
    """
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
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy construction with multiple seeds
    initial_scope = max(130, counting_bound + 25)
    
    # Deterministic first
    rows = _greedy_construct(n_rows, num_elements, random.Random(0), initial_scope)
    if rows is not None and _is_valid(rows):
        s = _scope(rows)
        best_rows = rows
        best_scope = s
    
    # Try randomized greedy with multiple seeds
    for seed in range(12):
        if time.time() > DEADLINE - 90:
            break
        rng_local = random.Random(seed * 17 + 3)
        rows = _greedy_construct(n_rows, num_elements, rng_local, initial_scope)
        if rows is not None and _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
    
    # Phase 2: CSP backtracking for tighter scopes
    if time.time() < DEADLINE - 70:
        start_scope = best_scope if best_rows else counting_bound + 30
        # Try a range of scopes
        for scope in range(counting_bound, min(start_scope + 1, counting_bound + 25)):
            if time.time() > DEADLINE - 50:
                break
            
            for attempt in range(2):
                if time.time() > DEADLINE - 50:
                    break
                
                rng_csp = random.Random(hash((scope, attempt, n_rows, k)) % (2**32))
                time_limit = time.time() + 3.0
                
                rows = _csp_backtrack(n_rows, num_elements, scope, rng_csp, time_limit)
                if rows is not None:
                    if _is_valid(rows):
                        s = _scope(rows)
                        if s < best_scope:
                            best_scope = s
                            best_rows = rows
                        break
    
    # If we still don't have a valid solution, use fallback
    if best_rows is None:
        best_rows = _fallback_construct(n_rows, num_elements)
        best_scope = _scope(best_rows)
    
    # Phase 3: Local search refinement
    if time.time() < DEADLINE - 30:
        improved_rows, improved_scope = _local_search_refine(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 35
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 4: Simulated annealing if time permits
    if time.time() < DEADLINE - 15:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 10,
            initial_temp=max(3.0, best_scope * 0.04)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        return _fallback_construct(n_rows, num_elements)
