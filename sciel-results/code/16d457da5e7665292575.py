import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimizes the largest entry (scope).
    """
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    lower_bound = total_diffs_needed
    
    # We use a global variable to pass the solution up
    global _solution_found
    _solution_found = None

    def solve_with_bound(M):
        global _solution_found
        _solution_found = None
        
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        def dfs(r, c):
            # Pruning: Check if enough differences are available
            # Calculate remaining differences to be generated
            rem_diffs_current_row = 0
            # We are about to fill indices c, c+1, ..., k
            # When filling index j, we generate j differences (with indices 0..j-1)
            for j in range(c, k + 1):
                rem_diffs_current_row += j
                
            num_future_rows = n_rows - 1 - r
            rem_diffs_future_rows = num_future_rows * (k * (k + 1) // 2)
            
            total_rem_diffs = rem_diffs_current_row + rem_diffs_future_rows
            
            # Available differences are integers in [1, M] not in used_diffs
            # Count = M - len(used_diffs)
            if M - len(used_diffs) < total_rem_diffs:
                return False

            if r == n_rows:
                _solution_found = [row[:] for row in rows]
                return True
                
            # Determine range for next value
            if c == 1:
                if r == 0:
                    start_val = 1
                else:
                    start_val = rows[r-1][1]
            else:
                start_val = rows[r][c-1] + 1
                
            # Max value must allow for remaining elements to be strictly increasing and <= M
            # We need to place (k - c) more elements after this one.
            # So val + (k - c) <= M
            max_val = M - (k - c)
            
            if start_val > max_val:
                return False

            for val in range(start_val, max_val + 1):
                # Check differences
                valid = True
                new_diffs = []
                for i in range(c):
                    d = val - rows[r][i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                    
                if not valid:
                    continue
                    
                # Commit
                used_diffs.update(new_diffs)
                rows[r].append(val)
                
                if c == k:
                    if dfs(r + 1, 1):
                        return True
                else:
                    if dfs(r, c + 1):
                        return True
                
                # Backtrack
                rows[r].pop()
                used_diffs.difference_update(new_diffs)
                
            return False

        return dfs(0, 1)

    # Iterative Deepening
    # Start from lower bound. 
    # The gap between lower bound and optimal can be small.
    # We search up to lower_bound + 1000 to be safe, but break early if found.
    # For n=7, k=5, lower_bound=105, optimal=112.
    
    # To ensure we don't hang if no solution exists in the range (shouldn't happen for valid inputs),
    # we cap the search.
    
    for M in range(lower_bound, lower_bound + 200):
        if solve_with_bound(M):
            return _solution_found
            
    # If not found, return empty list (invalid)
    return []
