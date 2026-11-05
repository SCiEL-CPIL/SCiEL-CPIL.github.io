import time

def construct(n_rows, k):
    """
    Finds a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct over all rows.
    Minimizes the largest entry (scope).
    """
    
    def run_greedy():
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
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = run_greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # Calculate theoretical lower bound for scope
    # Total distinct differences required: n_rows * C(k+1, 2)
    # These differences must be distinct integers from 1..Scope
    # So Scope * (Scope + 1) / 2 >= n_rows * C(k+1, 2)
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    lb = 1
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
    deadline = time.time() + 140.0
    current_target = best_scope - 1
    
    while current_target >= lb and time.time() < deadline:
        # Reset state for this search attempt
        used_diffs = set()
        rows_found = []
        search_exhaustive = True
        
        def backtrack(row_idx, col_idx, current_row):
            nonlocal search_exhaustive
            
            # Time check
            if time.time() > deadline:
                search_exhaustive = False
                return False
            
            if row_idx == n_rows:
                return True
            
            if col_idx == k:
                # Row is complete
                rows_found.append(list(current_row))
                return backtrack(row_idx + 1, 0, [0])
            
            prev_val = current_row[-1]
            
            # Heuristic: Try smallest valid values first
            # This helps in keeping the scope low
            for v in range(prev_val + 1, current_target + 1):
                new_diffs = {v - a for a in current_row}
                
                # Check for conflicts with global used differences
                if new_diffs & used_diffs:
                    continue
                
                # Commit
                used_diffs |= new_diffs
                current_row.append(v)
                
                if backtrack(row_idx, col_idx + 1, current_row):
                    return True
                
                # Undo
                current_row.pop()
                used_diffs -= new_diffs
            
            return False
        
        # Start the search
        found = backtrack(0, 0, [0])
        
        if found:
            best_rows = rows_found
            best_scope = current_target
            current_target -= 1
        else:
            if search_exhaustive:
                # We proved no solution exists for this scope.
                # Since scope must be an integer, no solution exists for smaller scopes either.
                break
            else:
                # Timed out. We don't know if a solution exists.
                # To be safe and fast, we stop.
                break

    return best_rows
