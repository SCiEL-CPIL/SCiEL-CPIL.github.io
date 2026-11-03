import time
import sys

def construct(n_rows, k):
    """
    Construct a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Minimizes the largest entry (scope).
    """
    
    # Fallback: Greedy construction (same as seed)
    def greedy_construct():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Get the baseline solution
    baseline_sol = greedy_construct()
    baseline_scope = max(max(r) for r in baseline_sol)
    
    # Lower bound on scope
    # Total number of differences is n_rows * C(k+1, 2)
    # These differences must be distinct integers >= 1
    # So max difference >= total_diffs
    total_diffs = n_rows * (k * (k + 1) // 2)
    lb = total_diffs
    
    start_time = time.time()
    deadline = start_time + 100.0  # Keep 20s margin for 120s limit
    
    # Backtracking solver
    def solve_with_limit(M, time_limit):
        """
        Try to find a valid DTS with max entry <= M.
        Returns the list of rows if found, None otherwise.
        """
        used = set()
        rows = []
        current_row = []
        
        # To avoid excessive copying, we modify in place and backtrack
        # We need to track the state of 'used' and 'rows' carefully.
        
        def backtrack(row_idx, pos_in_row):
            # Check time limit
            if time.time() - start_time > deadline:
                return False
            
            if row_idx == n_rows:
                if pos_in_row == 0:
                    return True
                else:
                    return False
            
            if pos_in_row == k + 1:
                rows.append(list(current_row))
                return backtrack(row_idx + 1, 0)
            
            # Determine the start value for the next element
            if pos_in_row == 0:
                # First element is always 0
                current_row.append(0)
                # No new differences for 0
                res = backtrack(row_idx, 1)
                current_row.pop()
                return res
            
            last_val = current_row[-1]
            # Try values from last_val + 1 up to M
            # Heuristic: Try smaller values first
            for x in range(last_val + 1, M + 1):
                # Check time
                if time.time() - start_time > deadline:
                    return False
                
                # Calculate new differences
                new_diffs = {x - a for a in current_row}
                
                # Check for conflicts with global used set
                if new_diffs & used:
                    continue
                
                # Place x
                current_row.append(x)
                used.update(new_diffs)
                
                if backtrack(row_idx, pos_in_row + 1):
                    return True
                
                # Backtrack
                current_row.pop()
                used.difference_update(new_diffs)
                
            return False
        
        # Start the search
        # We need to initialize current_row as empty
        # The backtrack function handles pos_in_row=0 by appending 0
        
        if backtrack(0, 0):
            return rows
        return None

    # Iterate through possible scopes
    # Start from lower bound, go up
    # Allocate a small time budget per scope value to ensure we explore multiple scopes
    # E.g., 2 seconds per scope
    time_per_scope = 2.0
    
    for M in range(lb, lb + 200):  # Explore up to 200 above LB
        if time.time() > deadline:
            break
            
        # Check if this M is worse than baseline
        # If M >= baseline_scope, we might still find a better solution?
        # No, if we found a solution with scope M, and M < baseline_scope, it's better.
        # If M >= baseline_scope, the score won't improve (unless baseline was invalid, which it isn't).
        # So we can stop if M >= baseline_scope, unless we haven't found anything better yet?
        # Actually, if we find a solution with M < baseline_scope, we return it immediately.
        # If we reach M = baseline_scope, we know the baseline is valid, so we can stop.
        if M >= baseline_scope:
            break
            
        sol = solve_with_limit(M, time_per_scope)
        if sol is not None:
            # Verify solution just in case
            # (Optional but good for safety)
            # Check all diffs distinct
            all_diffs = set()
            valid = True
            for r in sol:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            if valid:
                return sol
                
    return baseline_sol

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)
