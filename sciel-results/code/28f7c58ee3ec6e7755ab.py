import time

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    """
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin for 120s limit

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Get initial greedy solution
    best_rows = greedy_construct()
    best_scope = max(max(row) for row in best_rows) if best_rows else 0
    
    # Theoretical lower bound for scope
    # Total differences needed: n_rows * C(k+1, 2)
    # Max difference in a row with max entry M is M.
    # So M >= Total Diffs
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    min_scope = total_diffs_needed
    
    # If greedy is already at lower bound, return it
    if best_scope <= min_scope:
        return best_rows

    # Iterative Deepening DFS to find a solution with smaller scope
    # Try M from best_scope - 1 down to min_scope
    for M in range(best_scope - 1, min_scope - 1, -1):
        if time.time() - start_time > time_limit:
            break
            
        used_diffs = set()
        rows = []
        
        def dfs(r, c, current_row):
            # Time check
            if time.time() - start_time > time_limit:
                return None
            
            # Base case: all rows constructed
            if r == n_rows:
                # Return deep copy to avoid mutation issues
                return [row[:] for row in rows]
            
            # If current row is full, move to next row
            if c == k + 1:
                rows.append(current_row)
                # Start next row with 0
                return dfs(r + 1, 1, [0])
            
            # Find next number for current row
            last_val = current_row[-1]
            # Try candidates from last_val + 1 to M
            for x in range(last_val + 1, M + 1):
                # Check if differences are valid
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    # Apply choice
                    used_diffs.update(new_diffs)
                    current_row.append(x)
                    
                    # Recurse
                    res = dfs(r, c + 1, current_row)
                    if res is not None:
                        return res
                    
                    # Undo choice
                    current_row.pop()
                    used_diffs.difference_update(new_diffs)
            
            return None
        
        # Start search for this M
        # Row 0 starts with [0], so we need to fill indices 1 to k
        result = dfs(0, 1, [0])
        if result is not None:
            return result

    # If no better solution found, return greedy
    return best_rows
