import time
import random
import math

# EVOLVE-BLOCK-START

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

def _backtrack_solve_row(blocked_diffs, k, max_val):
    """
    Find a row of length k+1 starting at 0, strictly increasing, max entry <= max_val,
    such that its internal diffs are distinct and disjoint from blocked_diffs.
    Returns the row (list) or None.
    Uses DFS with forward checking, trying smallest values first to minimize max.
    """
    row = [0]
    current_diffs = set()
    
    # Pre-calculate min gap needed for remaining steps to ensure we don't overshoot max_val?
    # Actually, just checking v <= max_val at each step is sufficient if we are careful.
    # For step i (0-indexed, 0 is already set), we are choosing row[i].
    # We need k-i more elements after row[i].
    # The minimum possible value for the next element is row[i] + 1.
    # So if row[i] + (k-i) > max_val, it's impossible.
    
    def dfs(idx, last_val):
        if idx == k + 1:
            return row[:]
        
        # Determine range for next value
        # idx is the position we are filling (1 to k)
        # last_val is row[idx-1]
        
        # Minimum next val
        min_v = last_val + 1
        
        # Maximum next val
        # If this is the last element (idx == k), max is max_val
        # If not, we need space for (k - idx) more elements.
        # Each subsequent element must be at least +1 from the previous.
        # So max_v = max_val - (k - idx)
        
        if idx == k:
            max_v = max_val
        else:
            max_v = max_val - (k - idx)
            
        if min_v > max_v:
            return None
            
        # Try values from min_v to max_v
        # Optimization: We can limit the search range if we know we are looking for a specific max?
        # No, we just want ANY valid row with max <= max_val. Smallest is preferred for future flexibility.
        
        for v in range(min_v, max_v + 1):
            new_diffs = {v - u for u in row}
            
            # Check internal conflicts
            if new_diffs & current_diffs:
                continue
                
            # Check external conflicts
            if new_diffs & blocked_diffs:
                continue
                
            # Valid
            row.append(v)
            current_diffs.update(new_diffs)
            
            res = dfs(idx + 1, v)
            if res is not None:
                return res
                
            row.pop()
            current_diffs.difference_update(new_diffs)
            
        return None
        
    return dfs(1, 0)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = max(max(r) for r in rows)
    
    # 2. Iterated Row Re-solve (IRRS)
    stagnation = 0
    max_stagnation = 2000
    
    # Pre-compute row diffs for quick removal
    row_diffs = [_row_diffs_set(r) for r in rows]
    
    while time.time() < deadline:
        # Select a row to re-solve
        # Strategy: Pick the row with the largest max, or random.
        # Picking the largest max is more likely to reduce the global max.
        # But we should also explore others.
        if random.random() < 0.7:
            # Pick row with max value
            r = max(range(n_rows), key=lambda i: max(rows[i]))
        else:
            r = random.randrange(n_rows)
            
        old_row = rows[r]
        old_diffs = row_diffs[r]
        old_max = max(old_row)
        
        # Remove this row's diffs from global used
        all_used -= old_diffs
        
        # Determine target max for this row
        # We want to see if we can find a row with max < current global max
        # If this row IS the global max, we try to reduce it.
        # If not, we try to keep it same or reduce it.
        
        current_global_max = max(max(row) for row in rows)
        
        # Try to find a row with max <= old_max (to maintain validity)
        # But we prefer smaller.
        # Let's try to find a row with max < old_max first?
        # Or just max <= old_max. The DFS tries smallest first, so it will naturally find the smallest possible max.
        
        target_max = old_max
        
        new_row = _backtrack_solve_row(all_used, k, target_max)
        
        if new_row is not None:
            new_diffs = _row_diffs_set(new_row)
            new_max = max(new_row)
            
            # Check if this new row is better
            # Better means: new_max < old_max (local improvement)
            # OR: new_row != old_row (structural change, might help others)
            
            if new_max < old_max:
                # Update state
                rows[r] = new_row
                row_diffs[r] = new_diffs
                all_used |= new_diffs
                
                # Check global improvement
                new_global_max = max(max(row) for row in rows)
                if new_global_max < best_val:
                    best_val = new_global_max
                    best_rows = [r[:] for r in rows]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                # new_max == old_max
                # Only accept if we are stuck and the row is different (to allow exploration)
                if new_row != old_row and stagnation > max_stagnation // 2:
                    rows[r] = new_row
                    row_diffs[r] = new_diffs
                    all_used |= new_diffs
                    stagnation = 0 # Reset to allow more exploration
                else:
                    # Revert: put old diffs back
                    all_used |= old_diffs
                    # rows[r] is unchanged, row_diffs[r] unchanged
        else:
            # No valid row found? This should not happen if old_row was valid and we used old_max.
            # But if it does, revert.
            all_used |= old_diffs
            stagnation += 1
            
        # If stagnation is very high, restart from best
        if stagnation > max_stagnation:
            rows = [r[:] for r in best_rows]
            all_used = _compute_all_used(rows)
            row_diffs = [_row_diffs_set(r) for r in rows]
            stagnation = 0
            
    return best_rows

# EVOLVE-BLOCK-END
