<<<<<<< SEARCH
def _bitset_dfs(rows, k, max_bound, time_limit, rng):
    """
    Depth-first search using bitmasks to find a solution with max entry < max_bound.
    rows: list of lists, some elements may be -1 (unassigned).
    k: number of intervals (so k+1 elements per row).
    max_bound: strict upper bound for the largest entry.
    """
    L = k + 1
    n_rows = len(rows)
    used_mask = 0
    # Calculate initial used mask from assigned parts
    for r in range(n_rows):
        row = rows[r]
        # Only consider assigned elements (non -1)
        assigned = [x for x in row if x != -1]
        if len(assigned) > 1:
            # We need to be careful: if the row is partially assigned, 
            # the differences between assigned elements are fixed.
            # However, differences involving unassigned elements are not yet fixed.
            # So we just add diffs between currently assigned elements.
            for i in range(len(assigned)):
                for j in range(i+1, len(assigned)):
                    d = assigned[j] - assigned[i]
                    used_mask |= (1 << d)
    
    # Flatten the problem: we will assign elements in a specific order.
    # Order: Row 0 pos 1..L-1, Row 1 pos 1..L-1, etc.
    # But we must maintain increasing order within a row.
    
    # Precompute indices to visit
    order = []
    for r in range(n_rows):
        for pos in range(1, L):
            order.append((r, pos))
    
    # We need to track the last assigned value in each row to enforce increasing order
    last_val = [0] * n_rows # rows[r][0] is always 0
    for r in range(n_rows):
        if rows[r][0] != 0:
             return False # Should not happen
    
    # We need to know which positions are already assigned in the input
    # For simplicity, let's assume we start with empty rows except pos 0 for DFS.
    # If rows are partially filled, we handle it by skipping assigned positions.
    
    # Let's restart with empty rows for the DFS to be clean.
    # The caller will pass a fresh set of empty rows if it wants to use this.
    # But to be generic, let's handle the input as is.
    
    # Actually, it's safer to just use the DFS on empty rows.
    # Let's create a dedicated function that takes n_rows, k, max_bound.
    pass

def _row_diff_mask(row):
    """Compute the bitmask of differences for a single row."""
    mask = 0
    L = len(row)
    for i in range(L):
        for j in range(i + 1, L):
            d = row[j] - row[i]
            mask |= (1 << d)
    return mask

def _find_better_solution_bitset(n_rows, k, max_bound, time_limit, rng):
    """
    Attempt to find a solution with max entry < max_bound using bitset DFS.
    Returns a list of rows or None.
    """
    L = k + 1
    start_time = time.time()
    
    # We will assign elements row by row, position by position.
    # rows[r][pos]
    rows = [[0] + [-1] * (L - 1) for _ in range(n_rows)]
    used_mask = 0
    
    # Order of assignment: Row 0, then Row 1, etc.
    # Within a row, pos 1 to L-1.
    
    # To speed up, we can pre-compute candidate values for each position?
    # No, the constraints change as we assign.
    
    # Backtracking state:
    # current_row_idx, current_pos_idx
    # We use a recursive function with explicit stack or recursion limit increase.
    # Given L=9, n_rows=4, total depth 32. Recursion is fine.
    
    sys.setrecursionlimit(10000)
    
    def dfs(r_idx, pos_idx):
        nonlocal used_mask, rows
        
        if time.time() - start_time > time_limit:
            return False
        
        if r_idx == n_rows:
            return True # All rows assigned
        
        if pos_idx == L:
            # Move to next row
            return dfs(r_idx + 1, 1)
        
        # We are assigning rows[r_idx][pos_idx]
        # Lower bound: rows[r_idx][pos_idx - 1] + 1
        # Upper bound: max_bound - 1 (since we want < max_bound, and must leave room for remaining? 
        # Actually, if this is the last pos, upper is max_bound-1.
        # If not last, we must ensure we can fit remaining elements.
        # Simple bound: max_bound - 1. We can refine later.
        
        lower = rows[r_idx][pos_idx - 1] + 1
        upper = max_bound - 1
        
        # Additional constraint: We need to fit L - pos_idx - 1 more elements after this one.
        # So the value cannot be too large? No, it can be large, but the *next* ones must be larger.
        # The constraint is that the *last* element must be < max_bound.
        # So current value must be <= max_bound - (L - pos_idx - 1)
        # Because we need at least 1 increment for each remaining slot.
        upper = min(upper, max_bound - (L - pos_idx - 1))
        
        if lower > upper:
            return False
            
        # Candidates: lower to upper
        # We can shuffle candidates to add randomness? 
        # For a "find any solution" DFS, increasing order is often best for finding *a* solution.
        # But we want to find a solution with *small* max entry. 
        # Since we are searching for < max_bound, any solution is good.
        # Let's try increasing order first. If that fails, try random?
        # Let's stick to increasing for determinism in the "best effort" sense.
        
        for val in candidates:
            new_diffs_mask = 0
            for i in range(pos_idx):
                d = val - rows[r_idx][i]
                new_diffs_mask |= (1 << d)
            
            if new_diffs_mask & used_mask:
                continue
            
            # Accept val
            rows[r_idx][pos_idx] = val
            used_mask |= new_diffs_mask
            
            if dfs(r_idx, pos_idx + 1):
                return True
            
            # Backtrack
            used_mask ^= new_diffs_mask # XOR works because bits are unique in new_diffs_mask and not in used_mask before
            rows[r_idx][pos_idx] = -1
            
        return False
    
    if dfs(0, 1):
        return rows
    return None
=======
def _backtrack_gaps(n_rows, k, max_bound, time_limit, rng):
    """
    Backtracking search over gaps.
    Each row is defined by k gaps g_1..g_k, where v_i = sum_{j=1}^i g_j.
    v_0 = 0.
    All differences v_j - v_i must be distinct globally.
    """
    L = k + 1
    start_time = time.time()
    gaps = [[0] * k for _ in range(n_rows)]
    used_mask = 0
    
    # Precompute prefix sums for each row as we go
    # row_vals[r][i] = sum(gaps[r][0..i-1])
    
    def dfs(r_idx, g_idx):
        nonlocal used_mask
        
        if time.time() - start_time > time_limit:
            return False
        
        if r_idx == n_rows:
            return True
        
        if g_idx == k:
            # Move to next row
            return dfs(r_idx + 1, 0)
        
        # Current position in the row
        # Current value v_curr = sum(gaps[r_idx][0..g_idx-1])
        # We are choosing gap g = gaps[r_idx][g_idx]
        # New value v_new = v_curr + g
        
        # Calculate v_curr
        v_curr = sum(gaps[r_idx][:g_idx])
        
        # Bounds for g
        # v_new must be > v_curr (g >= 1)
        # v_new must be such that the final value v_last < max_bound
        # v_last = v_curr + g + sum(remaining gaps)
        # Min remaining gaps = 1 each
        # So v_curr + g + (k - g_idx - 1) * 1 < max_bound
        # g < max_bound - v_curr - (k - g_idx - 1)
        
        min_g = 1
        max_g = max_bound - v_curr - (k - g_idx - 1) - 1
        
        if max_g < min_g:
            return False
        
        # To add diversity, we can shuffle the range or try different orders
        # For now, let's try small gaps first (compact)
        candidates = range(min_g, max_g + 1)
        
        # Optimization: We can check if any difference formed by this gap is already used
        # Differences formed:
        # 1. g itself (v_new - v_curr)
        # 2. v_new - v_i for all i < g_idx (differences with previous elements in this row)
        # Note: Differences with future elements in this row are not known yet, but they will be checked when those gaps are assigned.
        # However, we must ensure that the difference g does not conflict with any previously used difference.
        # Also, v_new - v_i for i < g_idx must not conflict.
        
        # Precompute prefix sums for this row up to g_idx
        prefix = [0] * (g_idx + 1)
        for i in range(g_idx):
            prefix[i+1] = prefix[i] + gaps[r_idx][i]
        # v_i = prefix[i]
        # v_curr = prefix[g_idx]
        
        for g in candidates:
            if time.time() - start_time > time_limit:
                return False
            
            v_new = v_curr + g
            
            # Check conflicts
            # 1. g itself
            if (1 << g) & used_mask:
                continue
            
            # 2. v_new - v_i for i in 0..g_idx-1
            conflict = False
            for i in range(g_idx):
                d = v_new - prefix[i]
                if (1 << d) & used_mask:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Accept g
            gaps[r_idx][g_idx] = g
            
            # Update used_mask
            # Add g
            used_mask |= (1 << g)
            # Add v_new - v_i
            for i in range(g_idx):
                d = v_new - prefix[i]
                used_mask |= (1 << d)
            
            if dfs(r_idx, g_idx + 1):
                return True
            
            # Backtrack
            used_mask ^= (1 << g)
            for i in range(g_idx):
                d = v_new - prefix[i]
                used_mask ^= (1 << d)
            gaps[r_idx][g_idx] = 0
        
        return False
    
    if dfs(0, 0):
        # Convert gaps to rows
        rows = []
        for r in range(n_rows):
            row = [0]
            curr = 0
            for g in gaps[r]:
                curr += g
                row.append(curr)
            rows.append(row)
        return rows
    
    return None
>>>>>>> REPLACE

