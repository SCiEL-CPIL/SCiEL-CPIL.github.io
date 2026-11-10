import sys

def construct(n_rows, k):
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound for max value
    lower_bound = total_diffs
    
    # Search for the smallest M that allows a valid construction
    # We expand the range slightly to be safe, but the optimal is close to lower_bound
    # For (7,5), lower_bound is 105, optimal is 112.
    
    # We will try M from lower_bound to lower_bound + 50
    for M in range(lower_bound, lower_bound + 50):
        sol = solve_for_m(n_rows, k, M)
        if sol is not None:
            return sol
            
    # Fallback if not found in the expected range (should not happen for this task)
    # Try a larger range
    for M in range(lower_bound + 50, lower_bound + 200):
        sol = solve_for_m(n_rows, k, M)
        if sol is not None:
            return sol
            
    # Last resort: Greedy (may be invalid, but prevents crash)
    return greedy_fallback(n_rows, k)

def solve_for_m(n_rows, k, M):
    rows = []
    
    def backtrack_rows(row_idx, used_diffs):
        if row_idx == n_rows:
            return True
        
        row = [0]
        
        def backtrack_row(pos, row, used_diffs):
            if pos == k + 1:
                return True
            
            last_val = row[-1]
            
            # Heuristic: Try to find a valid x
            # We iterate x from last_val + 1 to M
            for x in range(last_val + 1, M + 1):
                new_used = used_diffs
                conflict = False
                for a in row:
                    diff = x - a
                    # Check if this difference is already used
                    if (new_used & (1 << diff)):
                        conflict = True
                        break
                    new_used |= (1 << diff)
                
                if conflict:
                    continue
                
                row.append(x)
                
                if backtrack_row(pos + 1, row, new_used):
                    return True
                
                row.pop()
            
            return False
        
        if backtrack_row(1, row, used_diffs):
            rows.append(row)
            
            # Calculate the final used_diffs to pass to the next row
            # We re-calculate from the row to ensure correctness, as 'used_diffs' 
            # in the closure is the state BEFORE this row was added.
            final_used = used_diffs
            for i in range(len(row)):
                for j in range(i):
                    final_used |= (1 << (row[i] - row[j]))
            
            if backtrack_rows(row_idx + 1, final_used):
                return True
            
            rows.pop()
        
        return False
    
    if backtrack_rows(0, 0):
        return rows
    return None

def greedy_fallback(n_rows, k):
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
