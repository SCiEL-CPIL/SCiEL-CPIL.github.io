import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_to_mask(row):
    """
    Convert a row to a bitmask of its differences.
    Also checks for internal duplicate differences.
    Returns (mask, is_valid).
    """
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    
    # Check for duplicates
    if len(diffs) != len(set(diffs)):
        return 0, False
        
    mask = 0
    for d in diffs:
        # d is at least 1
        mask |= (1 << d)
        
    return mask, True

def _generate_random_row(k, max_val, rng):
    """
    Generate a random strictly increasing row of length k+1 starting at 0,
    with max entry <= max_val.
    """
    # We need to choose k distinct integers from 1..max_val
    # Then sort them and prepend 0
    if max_val < k:
        return None
        
    # Sample k distinct integers
    # Using rng.sample is efficient
    try:
        chosen = rng.sample(range(1, max_val + 1), k)
    except ValueError:
        return None
        
    chosen.sort()
    row = [0] + chosen
    return row

def _find_independent_set(pool, n_rows, deadline):
    """
    Find a subset of n_rows items from pool with disjoint masks.
    pool: list of (row, mask)
    Uses backtracking.
    """
    # Sort pool by mask popcount? Or just random order?
    # To help symmetry breaking and search, we can sort by mask value or just keep as is.
    # Let's try sorting by mask to have a deterministic order, which might help pruning if we add it.
    # But random order might find a solution faster if the "easy" ones are scattered.
    # Let's stick to the order generated, but we can shuffle the pool before searching to vary the search path.
    
    n_pool = len(pool)
    if n_pool < n_rows:
        return None
        
    # Pre-compute conflicts? No, bitmask AND is fast enough.
    
    # Recursive search
    # To avoid deep recursion overhead, we can use an iterative stack or just recursion since depth is small (7).
    
    def search(idx, current_mask, start_idx):
        if idx == n_rows:
            return True
            
        if time.time() > deadline:
            return False
            
        # Heuristic: Try items from start_idx to end
        # We can also jump ahead if we detect we can't fill the rest
        # Simple bound: if n_pool - start_idx < n_rows - idx: return False
        
        for j in range(start_idx, n_pool):
            mask = pool[j][1]
            if current_mask & mask:
                continue
                
            # Try this item
            if search(idx + 1, current_mask | mask, j + 1):
                return True
                
        return False

    # To increase chances, we can shuffle the pool indices
    # But we need to return the actual rows.
    # Let's just run the search on the given pool order.
    # If it fails, the caller will generate a new pool.
    
    return search(0, 0, 0)

def _construct_with_pool(n_rows, k, M, deadline, rng):
    """
    Try to find a solution with max entry <= M using a random pool.
    """
    n_pool = 2000 # Size of the random pool
    
    # Generate pool
    pool = []
    seen_rows = set()
    
    for _ in range(n_pool * 2): # Generate more to fill pool with unique valid rows
        if time.time() > deadline:
            break
        if len(pool) >= n_pool:
            break
            
        row = _generate_random_row(k, M, rng)
        if row is None:
            continue
            
        # Quick check: row is sorted and starts with 0
        # Check for internal duplicates
        mask, valid = _row_to_mask(row)
        if not valid:
            continue
            
        # Avoid duplicate rows in pool
        row_tuple = tuple(row)
        if row_tuple in seen_rows:
            continue
        seen_rows.add(row_tuple)
        
        pool.append((row, mask))
        
    if len(pool) < n_rows:
        return None
        
    # Shuffle pool to randomize search order
    rng.shuffle(pool)
    
    # Run independent set search
    # We need to track which rows were selected.
    # The simple search above returns bool. We need to modify it to return the indices.
    
    selected = []
    
    def search_with_track(idx, current_mask, start_idx):
        if idx == n_rows:
            return True
            
        if time.time() > deadline:
            return False
            
        for j in range(start_idx, len(pool)):
            mask = pool[j][1]
            if current_mask & mask:
                continue
            
            selected.append(j)
            if search_with_track(idx + 1, current_mask | mask, j + 1):
                return True
            selected.pop()
            
        return False
        
    if search_with_track(0, 0, 0):
        return [pool[idx][0] for idx in selected]
        
    return None

def _greedy_fallback(n_rows, k):
    """
    Simple greedy fallback to ensure a valid result.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Best known lower bound is 105
    # Best known solution is 112
    # We search from 105 upwards
    
    best_solution = None
    best_scope = float('inf')
    
    # If k or n_rows is small, we might find it quickly
    # For (7, 5), we expect 112
    
    # We will try a few random pools for each M
    # If we find a solution, we return it immediately?
    # The task says "minimized", so we should check lower M first.
    
    # However, if we spend too much time on M=105 (which is likely impossible or hard), we might miss M=112.
    # Strategy: Allocate time budget to each M.
    # Or, just iterate and break if time is low.
    
    # Let's try M from 105 to 120
    # If we find a solution at M, we can try to improve it?
    # No, the first valid solution at the lowest M is optimal for that M.
    # We want the global minimum M.
    
    # To be safe, we'll run a few attempts for each M.
    
    rng = random.Random(42)
    
    for M in range(105, 130):
        if time.time() > deadline - 5:
            break
            
        # Try a few random pools for this M
        # If we find a solution, we have a candidate with scope <= M
        # We can stop if we are confident M is the minimum?
        # No, we must check if a smaller M was possible.
        # Since we iterate upwards, the first M that yields a solution is the minimum scope found.
        
        found = False
        # Try up to 20 random pools for this M
        for attempt in range(20):
            if time.time() > deadline - 2:
                break
                
            solution = _construct_with_pool(n_rows, k, M, deadline - 2, rng)
            if solution is not None:
                # Verify scope
                scope = max(max(r) for r in solution)
                if scope < best_scope:
                    best_scope = scope
                    best_solution = solution
                    
                # If we found a solution at M, and we are iterating from low to high,
                # this is the best possible for this M.
                # Can we do better? Only if we find a solution at M-1.
                # But we already tried M-1 and failed (or didn't try enough).
                # To be rigorous, we should continue searching lower M if we have time?
                # No, we iterate 105, 106...
                # If we found a solution at 112, we should check 111, 110... if we have time.
                # But usually, the first one found is the target.
                
                # Let's assume the first found is good enough, or we can spend remaining time trying M-1, M-2...
                # For now, return the first valid solution found in the ascending loop.
                return solution
        
    # If no solution found in the loop, use fallback
    if best_solution is not None:
        return best_solution
        
    return _greedy_fallback(n_rows, k)

# EVOLVE-BLOCK-END
