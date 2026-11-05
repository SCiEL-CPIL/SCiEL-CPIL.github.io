import time
import random
import math

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave buffer for evaluator
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Initial Construction using Randomized Greedy with Backtracking
    # We try multiple seeds.
    max_initial_seeds = 50
    for seed in range(max_initial_seeds):
        if time.time() > deadline - 30:
            break
        
        rows = _construct_row_by_row(n_rows, k, seed)
        if rows is None:
            continue
        
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
        if scope <= 105:
            break

    if best_rows is None:
        # Fallback: simple greedy
        return _simple_greedy(n_rows, k)

    # Phase 2: Local Search
    # We use a Hill Climbing approach with random restarts of rows.
    # Key fix: Ensure all moves maintain global and local validity.
    
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    
    rng = random.Random(12345)
    
    # Pre-compute global diffs for current state
    all_diffs = set()
    for r in current_rows:
        all_diffs.update(_row_diffs(r))
    
    while time.time() < deadline - 5:
        # Pick a random row to modify
        row_idx = rng.randint(0, n_rows - 1)
        
        # Calculate diffs of other rows
        other_diffs = set(all_diffs)
        current_row_diffs = _row_diffs(current_rows[row_idx])
        other_diffs.difference_update(current_row_diffs)
        
        # Try to rebuild the row to be "tighter" (smaller max element)
        # We use a randomized greedy that prefers smaller values but allows backtracking if stuck.
        new_row = _rebuild_row_safe(k, other_diffs, rng, target_max=current_rows[row_idx][-1] - 1)
        
        if new_row is None:
            # If we can't make it smaller, try to make it equal (random shuffle) to escape local optima
            new_row = _rebuild_row_safe(k, other_diffs, rng, target_max=current_rows[row_idx][-1])
        
        if new_row is not None:
            new_scope = max([r[-1] for r in current_rows[:row_idx] + [new_row] + current_rows[row_idx+1:]])
            
            if new_scope < current_scope:
                # Accept improvement
                current_rows[row_idx] = new_row
                current_scope = new_scope
                
                # Update global diffs
                all_diffs.difference_update(current_row_diffs)
                all_diffs.update(_row_diffs(new_row))
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                # With some probability, accept worse to escape local minima (Simulated Annealing style)
                # But we must be careful not to degrade too much.
                if rng.random() < 0.1: # 10% chance to accept worse
                    current_rows[row_idx] = new_row
                    current_scope = new_scope
                    all_diffs.difference_update(current_row_diffs)
                    all_diffs.update(_row_diffs(new_row))

    # Final Validation
    if not _is_valid(best_rows):
        # If somehow invalid, return the simple greedy which is always valid
        return _simple_greedy(n_rows, k)
        
    return best_rows

def _scope(rows):
    return max(r[-1] for r in rows)

def _row_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i+1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for r in rows:
        d = _row_diffs(r)
        if seen & d:
            return False
        seen.update(d)
    return True

def _construct_row_by_row(n_rows, k, seed):
    """
    Constructs rows one by one using a randomized greedy approach.
    Ensures global uniqueness of differences.
    """
    rng = random.Random(seed)
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        local_diffs = set()
        
        for _ in range(k):
            last_val = row[-1]
            found = False
            
            # Search for next element
            # We try small increments first, but allow random jumps
            max_search = 500
            for x in range(last_val + 1, last_val + max_search + 1):
                new_diffs = {x - a for a in row}
                
                # Check global uniqueness
                if new_diffs & used_diffs:
                    continue
                
                # Check local uniqueness (within the row being built)
                if new_diffs & local_diffs:
                    continue
                
                # Valid candidate
                row.append(x)
                local_diffs.update(new_diffs)
                used_diffs.update(new_diffs)
                found = True
                break
            
            if not found:
                return None
        
        rows.append(row)
        
    return rows

def _rebuild_row_safe(k, forbidden_diffs, rng, target_max=1000):
    """
    Rebuilds a single row of k+1 elements starting at 0,
    avoiding any difference in `forbidden_diffs` AND ensuring internal uniqueness.
    Returns None if it cannot construct a row with max element <= target_max.
    """
    row = [0]
    local_diffs = set()
    
    for _ in range(k):
        last_val = row[-1]
        candidates = []
        
        # Search for valid x
        # We prioritize smaller x to keep scope low
        # But we need to ensure we don't get stuck.
        
        # To speed up, we can limit the search range based on target_max
        # If we are at step i, we need to fit k-i elements in target_max - last_val
        # This is complex, so we just search up to target_max
        
        for x in range(last_val + 1, target_max + 1):
            new_diffs = {x - a for a in row}
            
            # Check against forbidden (other rows)
            if new_diffs & forbidden_diffs:
                continue
            
            # Check against local (current row)
            if new_diffs & local_diffs:
                continue
            
            # Valid
            candidates.append(x)
            
            # We don't break immediately; we collect a few to allow random choice
            # But for "safe" rebuild aiming for min scope, we might want the smallest.
            # However, for SA diversity, random is better.
            # Let's collect up to 10 candidates.
            if len(candidates) >= 10:
                break
        
        if not candidates:
            return None
            
        # Pick a candidate
        # Bias towards smaller values to help minimize scope
        # Weighted random: 1/x
        # Or just random from the first few.
        # Let's pick randomly from the collected candidates.
        x = rng.choice(candidates)
        
        row.append(x)
        local_diffs.update({x - a for a in row[:-1]})
        
    return row

def _simple_greedy(n_rows, k):
    """
    Fallback simple greedy. Always valid.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        local_diffs = set()
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and not (new_diffs & local_diffs):
                row.append(x)
                used.update(new_diffs)
                local_diffs.update(new_diffs)
            x += 1
        rows.append(row)
    return rows
