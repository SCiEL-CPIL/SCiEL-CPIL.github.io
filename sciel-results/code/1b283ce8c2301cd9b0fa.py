# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS).
    """
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # 1. Get a quick upper bound using greedy
    greedy_rows = _greedy_construct(n_rows, k)
    M_upper = max(max(row) for row in greedy_rows)
    
    # 2. Lower bound
    M_lower = total_diffs
    
    # 3. Search for optimal M
    # To save time, we can skip some values if we know they are impossible?
    # For now, iterate.
    
    # Optimization: If n_rows=4 and k=7, we know the answer is likely 134.
    # We can start from M_lower.
    
    for M in range(M_lower, M_upper + 1):
        # Check if M is feasible
        # A simple necessary condition: M >= total_diffs
        # Also, the max difference in a row of k+1 elements is at least k (if consecutive 0..k).
        # But we are constrained by M.
        
        result = _solve_dts(n_rows, k, M)
        if result is not None:
            return result
            
    # Fallback
    return greedy_rows

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    return rows

def _solve_dts(n_rows, k, M):
    """
    Solves DTS with max value M using backtracking.
    Returns list of rows or None.
    """
    rows = [[] for _ in range(n_rows)]
    used_diffs = set()
    
    # Symmetry breaking: 
    # The rows are interchangeable. We can enforce that the first row is lexicographically smallest.
    # Or, simpler: just solve.
    
    def backtrack(row_idx, pos_idx):
        if row_idx == n_rows:
            return True
            
        if pos_idx == k + 1:
            # Move to next row
            return backtrack(row_idx + 1, 0)
            
        current_row = rows[row_idx]
        last_val = current_row[-1] if current_row else 0
        
        # Pruning: 
        # If we are at the last position, we just need any valid val.
        # Otherwise, we need to ensure we don't run out of space.
        
        # Heuristic: Try values from low to high.
        # This tends to keep the row compact, which is good for packing.
        
        # Range of values
        start = last_val + 1
        end = M
        
        # Additional Pruning:
        # The smallest possible value for the next element is start.
        # The largest is end.
        # If we pick 'val', the differences generated are val - a for a in current_row.
        
        for val in range(start, end + 1):
            # Check conflicts
            conflict = False
            new_diffs = []
            for a in current_row:
                d = val - a
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
                
            if conflict:
                continue
                
            # Check if remaining differences can fit?
            # Simple bound: 
            # We have (k+1 - pos_idx) positions left in this row.
            # The max value is M.
            # The min value for the last element is at least val + (k - pos_idx) ? No, they don't have to be consecutive.
            
            # Place
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            if backtrack(row_idx, pos_idx + 1):
                return True
                
            # Undo
            current_row.pop()
            for d in new_diffs:
                used_diffs.remove(d)
                
        return False
        
    return None if not backtrack(0, 0) else rows

# EVOLVE-BLOCK-END
