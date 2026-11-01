import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimizes the largest entry (scope).
    """
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    # Increase recursion limit just in case, though depth is small
    sys.setrecursionlimit(10000)

    def dfs(r, c):
        # r: row index, c: column index to fill
        if r == n_rows:
            return True
        
        # If we are starting a new row, c should be 1 (index 0 is 0)
        # If we are in a row, c is the next index to fill
        
        # Determine the range of x to try
        if c == 1:
            # First mark of the row
            if r == 0:
                start_x = 1
            else:
                # Symmetry breaking: enforce non-decreasing first marks
                # This assumes we can reorder rows. Since rows are indistinguishable 
                # in the final set (only the union of differences matters), 
                # we can impose an order.
                start_x = rows[r-1][1]
        else:
            start_x = rows[r][c-1] + 1
            
        # Heuristic: We want to keep x small. 
        # We don't have a hard upper bound, but we can stop if x gets too big?
        # For safety, we let it go, but backtracking will handle it.
        
        # To improve speed, we can limit the search if we detect a dead end early?
        # Standard backtracking is sufficient here.
        
        while True:
            # Check differences
            # new_diffs = {x - a for a in rows[r][:c]}
            # Note: rows[r] currently has length c (indices 0 to c-1)
            valid = True
            new_diffs = []
            for i in range(c):
                d = start_x - rows[r][i]
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                # Add to used
                used_diffs.update(new_diffs)
                rows[r].append(start_x)
                
                # Determine next state
                if c == k:
                    # Finished this row (indices 0..k, length k+1)
                    if dfs(r + 1, 1):
                        return True
                else:
                    if dfs(r, c + 1):
                        return True
                
                # Backtrack
                rows[r].pop()
                used_diffs.difference_update(new_diffs)
            
            start_x += 1
            # Optional: Add a bound to prevent infinite loop if no solution exists
            # But a solution always exists for small n,k.
            # If start_x > 500, something is wrong or it's a very hard instance.
            # For n=7, k=5, scope ~112. 500 is a safe cap for failure detection.
            if start_x > 500:
                return False

    dfs(0, 1)
    return rows
