import math
import time
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct over all rows.
    Minimize the largest entry (scope).
    
    Uses backtracking with forward checking, symmetry breaking, and efficient difference tracking.
    """
    # Total number of differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Calculate a reasonable lower bound for the scope.
    # The number of distinct differences in [0, m] is m*(m+1)/2.
    # We need at least total_diffs distinct differences.
    # m*(m+1)/2 >= total_diffs  =>  m^2 + m - 2*total_diffs >= 0
    # m >= (-1 + sqrt(1 + 8*total_diffs)) / 2
    lb_diff = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    
    # The scope must also be at least k (since the row has k+1 elements 0, ..., x, x>=k)
    # Actually, the smallest possible scope for a single row of size k+1 is k (0, 1, ..., k).
    # So lb = max(lb_diff, k)
    lb = max(lb_diff, k)
    
    # In practice, the counting bound is often a lower bound, but the actual minimum scope
    # for difference triangle sets is often higher. We will search upwards from lb.
    
    # To handle the specific case where the counting bound is very low but the real solution
    # is much higher (like n=7, k=5 where lb_diff~14 but solution is 112), 
    # we might need to search a wide range. 
    # However, backtracking from a very low scope will fail quickly due to pruning.
    # The main bottleneck is when the scope is close to the optimal.
    
    def solve(scope):
        """
        Try to construct n_rows rows with max entry <= scope.
        Returns the list of rows if successful, else None.
        """
        # used_diffs[d] is True if difference d has been used
        # Max difference is scope
        used_diffs = [False] * (scope + 1)
        rows = []
        
        # Symmetry breaking: We can assume the rows are sorted by their last element.
        # Let last_max[i] be the last element of row i. We enforce last_max[i] <= last_max[i+1].
        # This reduces the search space by a factor of n_rows!
        
        def backtrack(row_idx, current_row, current_max, prev_last_max):
            """
            row_idx: current row index (0 to n_rows-1)
            current_row: list of elements in the current row (starts with [0])
            current_max: the last element added to current_row
            prev_last_max: the last element of the previous row (for symmetry breaking)
                           -1 if this is the first row
            """
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                # Symmetry check: current_row[-1] must be >= prev_last_max
                # But we are constructing rows in order, so we just need to ensure
                # that the last element of this row is <= the last element of the next row.
                # We don't know the next row's last element yet.
                # Standard symmetry breaking: enforce non-decreasing order of the last elements.
                if prev_last_max > current_row[-1]:
                    return False
                
                rows.append(current_row[:])
                # Move to next row
                if backtrack(row_idx + 1, [0], 0, current_row[-1]):
                    return True
                rows.pop()
                return False
            
            # Determine bounds for the next value x
            min_next = current_max + 1
            # Max next value is scope
            # Also, if this is not the last row, we might need to leave room for future rows?
            # No, future rows can have larger elements.
            # But due to symmetry breaking (rows sorted by last element), 
            # if this is the last row, current_row[-1] can be up to scope.
            # If this is not the last row, current_row[-1] must be <= scope, 
            # but also it must be possible to construct the remaining rows with last elements >= current_row[-1].
            
            max_next = scope
            
            # Heuristic: Try values that are "central" or leave more space?
            # Usually, for difference sets, greedy (smallest first) works well with pruning.
            # However, sometimes a slightly larger value allows a better packing.
            # We stick to increasing order for simplicity and determinism, 
            # relying on the symmetry breaking and pruning.
            
            for x in range(min_next, max_next + 1):
                # Check if adding x creates any duplicate differences
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d > scope:
                        # This shouldn't happen if x <= scope and a >= 0, but just in case
                        valid = False
                        break
                    if used_diffs[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Mark diffs as used
                for d in new_diffs:
                    used_diffs[d] = True
                
                current_row.append(x)
                
                if backtrack(row_idx, current_row, x, prev_last_max):
                    return True
                
                # Undo
                current_row.pop()
                for d in new_diffs:
                    used_diffs[d] = False
            
            return False
        
        return backtrack(0, [0], 0, -1)

    start_time = time.time()
    timeout = 115  # Leave 5s margin for Python overhead and return
    
    # We search for the minimal scope.
    # The counting bound lb is a lower bound.
    # We iterate scope from lb upwards.
    # To avoid getting stuck in a very bad scope for too long, we can limit the time per scope,
    # but backtracking usually fails fast for infeasible scopes due to pruning.
    # The danger is if the optimal scope is much larger than lb.
    # For n=7, k=5, lb is ~14, optimal is 112. 
    # Searching 14..111 will fail quickly?
    # Let's check: for scope=14, max diffs is 14*15/2 = 105. Total diffs needed is 7*15=105.
    # So scope=14 is the absolute minimum by counting.
    # But it's likely infeasible.
    # The search will try to fill the space.
    # With symmetry breaking, the search space is reduced.
    
    # Optimization: If lb is small, we might want to jump to a more reasonable estimate.
    # However, the problem asks to minimize the scope. We must find the true minimum.
    # If the true minimum is 112, we must check 14..111.
    # If each check takes 0.1s, 100 checks take 10s. This is fine.
    # If each check takes 1s, 100 checks take 100s. This is risky.
    
    # To mitigate risk, we can implement a time limit per scope or a global time limit.
    # If we run out of time, we return the best valid solution found so far (if any) or the last attempted scope's result (if valid).
    # But we must return a valid solution.
    # If we don't find a solution by the time limit, we can fall back to a greedy construction which always works (though scope might be large).
    
    best_solution = None
    
    for scope in range(lb, lb + 200): # Search up to 200 above the lower bound
        if time.time() - start_time > timeout:
            break
        
        result = solve(scope)
        if result is not None:
            # We found a valid solution for this scope.
            # Since we iterate scope upwards, the first valid one is the minimal.
            return result
        
    # Fallback: Greedy construction
    # This ensures we always return a valid solution, even if the scope is not minimal.
    # The score is based on the scope, so a large scope is bad, but better than 0 (timeout/invalid).
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
