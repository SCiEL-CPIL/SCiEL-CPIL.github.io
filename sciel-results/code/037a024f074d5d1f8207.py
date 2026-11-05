import time
import random
import math
import sys

def _row_diffs_list(row):
    """Return list of all positive differences within a row."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _row_mask(row):
    """Bitmask of differences in a row."""
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
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))

def _compute_all_diffs(rows):
    """Return (global_mask, row_masks) for all rows."""
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks

def _greedy_construct(n_rows, num_elements, rng, max_scope_hint=None):
    """
    Greedy construction: build rows one by one, picking the smallest valid value
    for each position with some randomization.
    """
    rows = []
    used_mask = 0
    
    for row_idx in range(n_rows):
        row = [0]
        for pos in range(1, num_elements):
            found = False
            if max_scope_hint:
                high = max_scope_hint
            else:
                high = 200
            
            candidates = list(range(row[-1] + 1, high + 1))
            
            if len(candidates) > 30:
                sample_size = min(25, len(candidates))
                weights = [1.0 / (1.0 + 0.5 * i) for i in range(len(candidates))]
                selected = []
                pool = list(range(len(candidates)))
                for _ in range(sample_size):
                    total_w = sum(weights[i] for i in pool)
                    r = rng.random() * total_w
                    cum = 0
                    for idx, pi in enumerate(pool):
                        cum += weights[pi]
                        if r <= cum:
                            selected.append(candidates[pi])
                            pool.remove(pi)
                            break
                candidates = sorted(selected)
            
            for val in candidates:
                new_diffs = {val - prev for prev in row}
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    continue
                existing_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        existing_diffs.add(row[j] - row[i])
                if new_diffs & existing_diffs:
                    continue
                
                row.append(val)
                used_mask |= new_mask
                found = True
                break
            
            if not found:
                row.append(row[-1] + 10 + rng.randint(0, 20))
                new_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        new_diffs.add(row[j] - row[i])
                for d in new_diffs:
                    used_mask |= (1 << d)
        
        rows.append(row)
    
    return rows

def _csp_backtrack(n_rows, num_elements, max_scope, used_mask, rng, deadline, node_limit, nodes):
    """
    Recursive backtracking with forward checking.
    Variables: (row_idx, pos_idx)
    Domain: values from current_pos+1 to max_scope
    Constraint: differences must be unique globally and within row.
    """
    if nodes[0] >= node_limit:
        return None
    nodes[0] += 1
    
    if time.time() > deadline:
        return None
    
    # Find the next variable to assign
    # We fill row by row, position by position for simplicity in this structural change
    # But we can use MRV if we track domains. For now, sequential is the base, 
    # but we add a "jump" heuristic: if a row is stuck, try reordering rows?
    # Actually, a true CSP approach would treat all n_rows * (num_elements-1) variables.
    # Let's stick to a robust sequential backtrack with aggressive pruning.
    
    # Determine which row and position we are filling
    # We pass state as list of rows, partially filled
    # To make this efficient, we need to know where we are.
    # Let's restructure: this function is called recursively.
    # State: rows (list of lists), current_row, current_pos
    
    # This specific implementation is a helper for the main driver.
    # The main driver will manage the recursion stack or call this recursively.
    
    pass

def _solve_csp(n_rows, num_elements, max_scope, rng, deadline, node_limit=500000):
    """
    Solve the difference triangle problem using CSP backtracking.
    """
    rows = [[0] for _ in range(n_rows)]
    used_mask = 0
    nodes = [0]
    
    def backtrack(row_idx, pos_idx):
        if nodes[0] >= node_limit or time.time() > deadline:
            return False
        
        nodes[0] += 1
        
        # If we are done with all rows
        if row_idx == n_rows:
            return True
        
        # If we are done with current row, move to next
        if pos_idx == num_elements:
            return backtrack(row_idx + 1, 1)
        
        current_row = rows[row_idx]
        last_val = current_row[-1]
        
        # Generate candidates for current position
        # Heuristic: Try values that leave room for the rest of the row
        # Min value is last_val + 1
        # Max value is max_scope - (num_elements - pos_idx - 1)
        min_val = last_val + 1
        max_val = max_scope - (num_elements - pos_idx - 1)
        
        if min_val > max_val:
            return False
            
        # Randomize candidate order slightly to escape local traps, 
        # but keep a bias towards smaller values to minimize scope.
        candidates = list(range(min_val, max_val + 1))
        if len(candidates) > 10:
            # Shuffle but keep first few small ones likely
            rng.shuffle(candidates)
            # Sort by value with some noise
            candidates.sort(key=lambda x: x + rng.random() * 5)
        
        for val in candidates:
            # Check validity of adding val to current_row
            new_diffs = set()
            valid = True
            for prev in current_row:
                d = val - prev
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            new_mask = 0
            for d in new_diffs:
                new_mask |= (1 << d)
            
            if new_mask & used_mask:
                continue
            
            # Accept and recurse
            current_row.append(val)
            new_used_mask = used_mask | new_mask
            
            if backtrack(row_idx, pos_idx + 1):
                return True
            
            # Backtrack
            current_row.pop()
            
        return False
    
    # Start with row 0, pos 1 (since pos 0 is always 0)
    if backtrack(0, 1):
        return rows
    return None

def _local_search_reduce(rows, num_elements, rng, time_limit):
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
    
    max_iter = 100000
    stagnation = 0
    max_stagnation = 5000
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break
        
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope
        
        if low >= old_val:
            continue
        
        new_val = rng.randint(low, old_val - 1)
        
        # Check validity
        row[pos] = new_val
        diffs_in_row = set()
        valid = True
        for i in range(n_elem):
            for j in range(i + 1, n_elem):
                d = row[j] - row[i]
                if d in diffs_in_row:
                    valid = False
                    break
                diffs_in_row.add(d)
            if not valid:
                break
        
        if not valid:
            row[pos] = old_val
            stagnation += 1
            continue
        
        new_row_mask = 0
        for d in diffs_in_row:
            new_row_mask |= (1 << d)
        
        other_mask = global_mask & ~row_masks[row_idx]
        if new_row_mask & other_mask:
            row[pos] = old_val
            stagnation += 1
            continue
        
        # Accept
        current_rows[row_idx][pos] = new_val
        old_row_mask = row_masks[row_idx]
        row_masks[row_idx] = new_row_mask
        global_mask = (global_mask & ~old_row_mask) | new_row_mask
        
        new_scope = _scope(current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in current_rows]
            stagnation = 0
        else:
            stagnation += 1
    
    return best_rows, best_scope

def _construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 135.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: CSP Backtracking
    # Try to find a solution with scope close to the counting bound.
    # We start from a reasonable upper bound and go down.
    # The counting bound is a lower bound on the max difference, but scope is max value.
    # Max difference <= scope. So scope >= counting_bound is a loose lower bound.
    # Actually, if max diff is D, scope can be up to D + (k-1) roughly? No, scope is the max value.
    # If we have values 0..S, max diff is S. So S >= counting_bound.
    
    # Let's try scopes from counting_bound + 10 downwards? No, we need to find A solution.
    # Let's try a few scopes upwards from counting_bound to ensure we find one, then refine.
    # Or just try a fixed number of CSP restarts with a tight scope.
    
    # Strategy: Try CSP with scope = counting_bound + offset for various offsets.
    for offset in [0, 2, 5, 10, 20, 50]:
        target_scope = counting_bound + offset
        if time.time() > DEADLINE - 30:
            break
            
        # Multiple restarts for each scope
        for restart in range(5):
            if time.time() > DEADLINE - 30:
                break
            
            rng_local = random.Random(restart * 13 + offset)
            rows = _solve_csp(n_rows, num_elements, target_scope, rng_local, DEADLINE - 30, node_limit=200000)
            
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                
                # If we found a solution at the lowest possible scope (counting bound), we are done with construction
                if target_scope == counting_bound:
                    break
        
        if best_rows is not None and best_scope <= counting_bound + 5:
            break
        
    # Phase 2: Fallback to Greedy if CSP fails
    if best_rows is None or best_scope > counting_bound + 20:
        for seed in range(10):
            if time.time() > DEADLINE - 60:
                break
            rng_local = random.Random(seed * 7 + 1)
            rows = _greedy_construct(n_rows, num_elements, rng_local)
            if _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
    
    # If no valid solution found, use a simple sequential fallback
    if best_rows is None:
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
        best_rows = rows
        best_scope = _scope(best_rows)
    
    # Phase 3: Local search refinement
    if time.time() < DEADLINE - 40:
        improved_rows, improved_scope = _local_search_reduce(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 30
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback: sequential construction
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
