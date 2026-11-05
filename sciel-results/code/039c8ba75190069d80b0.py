import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry using a backtracking CSP approach.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer
    
    # Lower bound for the scope (largest entry)
    # Total distinct differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    min_M = total_diffs
    
    def dfs(row_idx, pos, current_row, used_diffs, M, rows):
        # Check time
        if time.time() - start_time > time_limit:
            return False
            
        if pos == k + 1:
            # Row complete
            rows.append(current_row[:])
            
            if row_idx == n_rows - 1:
                return True
            
            # Start next row
            if dfs(row_idx + 1, 1, [0], used_diffs, M, rows):
                return True
            
            rows.pop()
            return False
        
        last_val = current_row[-1]
        candidates = []
        
        # Generate candidates x > last_val
        for x in range(last_val + 1, M + 1):
            new_diffs = []
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                # Heuristic: Count valid options for the next position (freedom)
                freedom = 0
                if pos + 1 < k + 1:
                    temp_row = current_row + [x]
                    temp_used = used_diffs.union(new_diffs)
                    last_val_next = temp_row[-1]
                    
                    for y in range(last_val_next + 1, M + 1):
                        ok = True
                        for a in temp_row:
                            if (y - a) in temp_used:
                                ok = False
                                break
                        if ok:
                            freedom += 1
                            if freedom > 20:  # Cap to save time
                                break
                
                candidates.append((freedom, x, new_diffs))
        
        if not candidates:
            return False
        
        # Sort by freedom (descending), then x (ascending)
        candidates.sort(key=lambda item: (-item[0], item[1]))
        
        for _, x, new_diffs in candidates:
            new_row = current_row + [x]
            new_used = used_diffs.union(new_diffs)
            
            if dfs(row_idx, pos + 1, new_row, new_used, M, rows):
                return True
        
        return False

    # Try to find solution for increasing M
    # Search a reasonable range above the lower bound
    for M in range(min_M, min_M + 150):
        if time.time() - start_time > time_limit:
            break
            
        rows = []
        # Start with first row, first element 0 already placed, so pos=1
        if dfs(0, 1, [0], set(), M, rows):
            return rows
            
    # Fallback to deterministic greedy if CSP search fails or times out
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
