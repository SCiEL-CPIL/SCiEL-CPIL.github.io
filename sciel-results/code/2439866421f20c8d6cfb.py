import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are globally distinct.
    Minimize the largest entry.
    
    Uses a backtracking search with symmetry breaking (row ordering by max value)
    and tight pruning based on a known upper bound.
    """
    start_time = time.time()
    max_time = 115.0
    
    best_rows = None
    best_scope = 120 # Known bound is 112, start a bit higher to find it quickly
    
    # We want to find a set of rows R_1, ..., R_n
    # Such that max(R_i) is minimized.
    # Symmetry breaking: max(R_1) <= max(R_2) <= ... <= max(R_n)
    
    def solve():
        nonlocal best_rows, best_scope, start_time, max_time
        
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row, min_max_val):
            """
            row_idx: index of the row being constructed (0 to n_rows-1)
            current_row: the list of values in the current row being built
            min_max_val: the minimum value the last element of this row must have
                         (to maintain non-decreasing order of row maxima)
            """
            nonlocal best_rows, best_scope, start_time, max_time
            
            if time.time() - start_time > max_time:
                return False
            
            # If we are starting a new row
            if len(current_row) == 0:
                current_row = [0]
            
            # If we have completed the current row
            if len(current_row) == k + 1:
                row_max = current_row[-1]
                
                # Check symmetry breaking constraint
                if row_max < min_max_val:
                    return False
                
                # If this is the last row, we have a complete solution
                if row_idx == n_rows - 1:
                    scope = row_max
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [r[:] for r in rows] + [current_row]
                    return True
                
                # Otherwise, commit this row and move to the next
                rows.append(current_row)
                # The next row must have a max >= current row's max
                success = backtrack(row_idx + 1, [], row_max)
                rows.pop()
                
                # If we found a better solution in the deeper recursion, 
                # we might want to continue searching for even better ones?
                # Since we are minimizing, and we update best_scope, 
                # we can prune if current row_max >= best_scope.
                # But we already checked row_max < min_max_val? No, min_max_val is lower bound.
                # We should prune if row_max >= best_scope because any subsequent row 
                # will have max >= row_max, so total scope >= best_scope.
                if row_max >= best_scope:
                    return False
                
                return success
            
            # Determine the next element in the current row
            last_val = current_row[-1]
            
            # Upper bound for the next value:
            # 1. It must be < best_scope (if we want to improve)
            # 2. It must be >= min_max_val? No, min_max_val applies to the LAST element.
            #    However, if we are at the last position (len == k), the candidate is the max.
            #    If we are not at the last position, the max will be larger.
            
            # Pruning: If we are at the last position, cand must be < best_scope.
            # If we are not at the last position, we need room for (k - len(current_row)) more steps.
            # Min increase is 1. So max_val >= cand + (k - len(current_row)).
            # Thus cand <= best_scope - 1 - (k - len(current_row)).
            
            pos = len(current_row)
            if pos == k:
                max_cand = best_scope - 1
            else:
                remaining_steps = k - pos
                max_cand = best_scope - 1 - remaining_steps
            
            # Also, if this is the last row, the max value is the total scope.
            # If row_idx == n_rows - 1, we just want to minimize this row's max.
            
            # Heuristic: Try candidates in ascending order.
            # Start from last_val + 1.
            # We can also restrict the search window.
            
            # Optimization: If we have a tight best_scope, the search space is small.
            # If best_scope is large, we might need to search wider.
            
            # Let's iterate candidates.
            # To speed up, we can check if the candidate creates a conflict.
            
            for cand in range(last_val + 1, max_cand + 1):
                if time.time() - start_time > max_time:
                    return False
                
                # Check differences
                new_diffs = set()
                conflict = False
                for a in current_row:
                    d = cand - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Additional Pruning: 
                # If this is the last element of the row, and row_idx < n_rows-1,
                # then cand is the max of this row.
                # If cand >= best_scope, we can't improve (already handled by max_cand).
                
                # Tentatively add
                current_row.append(cand)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, current_row, min_max_val):
                    return True
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        # Start with the first row. min_max_val is 0.
        # We can try to fix the first row to a "standard" small set to help symmetry breaking?
        # No, let the solver find it.
        
        backtrack(0, [], 0)
        
        # If no solution found with scope < 120, try a looser bound or fallback
        if best_rows is None:
            # Fallback to a simple greedy to ensure we return *something* valid
            # Though the backtracker should find something < 120 for (7,5)
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                rows.append(row)
            return rows

        return best_rows

    # Run the solver
    # The solver is deterministic with the current heuristic (ascending candidates).
    # To explore different branches if the first path fails or to find a better one,
    # we can run multiple instances with slightly different initial bounds or randomizations.
    # However, for (7,5), the optimal is 112. If we start with 120, the ascending search
    # will likely find 112 quickly if it exists.
    
    # Let's add a randomization restart loop just in case.
    # But first, try the deterministic run.
    
    solve()
    
    if best_rows is None:
        # If the tight search failed, do a broader search
        best_scope = 200
        solve()
        
    if best_rows is None:
        # Absolute fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_rows
