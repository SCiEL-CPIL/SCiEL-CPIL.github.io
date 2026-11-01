import time

def construct(n_rows, k):
    # Calculate lower bound
    total_diffs = n_rows * (k * (k + 1) // 2)
    M_min = total_diffs
    
    start_time = time.time()
    max_time = 110 # Leave buffer for interpreter overhead
    
    # Try to find a solution with backtracking
    # We try M from M_min upwards
    # Limit the number of M values to try to avoid infinite loop
    max_M_attempts = 50
    
    for M_offset in range(max_M_attempts):
        M = M_min + M_offset
        
        if time.time() - start_time > max_time:
            break
            
        rows_found = None
        
        # Define the solver for a specific M
        def solve_for_M(M_val):
            rows_local = []
            
            def bt(row_idx, used_diffs, prev_row):
                if row_idx == n_rows:
                    return True
                
                marks = [0]
                
                def fill(marks, row_diffs):
                    if len(marks) == k + 1:
                        # Check lex order
                        if row_idx > 0:
                            if not (marks > prev_row):
                                return False
                        
                        rows_local.append(marks)
                        
                        # Recurse
                        if bt(row_idx + 1, used_diffs | row_diffs, marks):
                            return True
                            
                        rows_local.pop()
                        return False
                    
                    last = marks[-1]
                    
                    # Optimization: 
                    # If we are trying to be lex greater than prev_row,
                    # and the current prefix is already strictly greater, we are good.
                    # If it is equal so far, the next mark must be >= prev_row's corresponding mark.
                    # If it is smaller, we must pick a mark strictly greater than prev_row's mark to "overtake".
                    
                    min_x = last + 1
                    if row_idx > 0:
                        # prev_row has length k+1
                        # Current index in marks is len(marks)-1
                        # We are choosing marks[len(marks)]
                        # prev_row index is len(marks)
                        if len(marks) < len(prev_row):
                            # If current prefix is smaller than prev_row prefix so far,
                            # we MUST pick x > prev_row[len(marks)]
                            # If current prefix is equal, we MUST pick x >= prev_row[len(marks)]
                            # If current prefix is greater, we can pick any x > last
                            
                            # Compare prefix
                            # marks is [0, ..., last]
                            # prev_row is [0, ..., prev_mark]
                            # We know marks[0] == prev_row[0] == 0
                            # For i in 1..len(marks)-1, marks[i] vs prev_row[i]
                            
                            # Simple check:
                            # If marks < prev_row[:len(marks)], we need x > prev_row[len(marks)]
                            # If marks == prev_row[:len(marks)], we need x >= prev_row[len(marks)]
                            
                            # Python list comparison handles this, but we need the bound for x.
                            # Let's just use the general case for simplicity, 
                            # the lex check at the end will prune invalid branches.
                            # However, adding a lower bound on x can help.
                            
                            # If marks < prev_row[:len(marks)], then x must be > prev_row[len(marks)]
                            # If marks == prev_row[:len(marks)], then x must be >= prev_row[len(marks)]
                            
                            # To keep it simple and robust, I'll skip this specific pruning 
                            # unless the search is too slow. The final check is sufficient.
                            pass
                    
                    for x in range(min_x, M_val + 1):
                        new_diffs = {x - a for a in marks}
                        if new_diffs & (used_diffs | row_diffs):
                            continue
                        
                        marks.append(x)
                        if fill(marks, row_diffs | new_diffs):
                            return True
                        marks.pop()
                        
                    return False
                
                return fill(marks, set())
            
            if bt(0, set(), None):
                return rows_local
            return None

        rows_found = solve_for_M(M)
        if rows_found:
            return rows_found

    # Fallback to greedy if backtracking fails or times out
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
