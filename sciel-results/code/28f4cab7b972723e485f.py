def _backtrack_solve(n_rows, k, M):
    used_diffs = set()
    rows = [[0] for _ in range(n_rows)] # Each row starts with 0
    
    def backtrack(row_idx, pos):
        # pos is the index of the element we are about to fill in rows[row_idx]
        # So we are looking for rows[row_idx][pos]
        
        if row_idx == n_rows:
            return True
        
        # If pos == k+1, the row is full. Move to next row.
        if pos == k + 1:
            return backtrack(row_idx + 1, 1)
        
        current_row = rows[row_idx]
        prev_val = current_row[-1]
        
        for val in range(prev_val + 1, M + 1):
            new_diffs = []
            conflict = False
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if conflict:
                continue
            
            # Add diffs
            for d in new_diffs:
                used_diffs.add(d)
            
            # Place value
            current_row.append(val)
            
            if backtrack(row_idx, pos + 1):
                return True
            
            # Backtrack
            current_row.pop()
            for d in new_diffs:
                used_diffs.remove(d)
                
        return False

    if backtrack(0, 1):
        return [r[:] for r in rows]
    return None
