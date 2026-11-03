import time
import sys

def construct(n_rows, k):
    # n_rows = 7, k = 5 -> 6 elements per row
    # Target: minimize max element
    # Known best: 112
    # Bound: 105
    
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Total differences per row: k*(k+1)/2 = 5*6/2 = 15
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound for M
    # M must be at least total_diffs_needed
    M_min = total_diffs_needed
    
    # We will search for M from M_min upwards
    # We keep the best solution found
    best_solution = None
    best_M = float('inf')
    
    # Helper to check if a solution is valid (for safety)
    def is_valid_solution(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def dfs(row_idx, col_idx, current_row, used_diffs, M, node_limit, nodes_visited):
        # nodes_visited is a list to allow modification in recursion
        nodes_visited[0] += 1
        if nodes_visited[0] > node_limit:
            return None, False
        
        if time.time() - start_time > time_limit:
            return None, False
            
        if row_idx == n_rows:
            # All rows constructed
            return rows_copy, True # Need to pass rows list or reconstruct
        
        # Pruning: Check if enough differences are available
        # Diffs used so far: len(used_diffs)
        # Diffs needed for current row: 
        # current_row has col_idx elements (if col_idx is the next index to fill, then current_row has col_idx elements? 
        # Let's define col_idx as the index of the element to be placed.
        # So current_row has length col_idx.
        # Diffs already in current_row: col_idx * (col_idx - 1) // 2
        diffs_in_current = col_idx * (col_idx - 1) // 2
        diffs_needed_current = diffs_per_row - diffs_in_current
        
        # Diffs needed for remaining rows (excluding current)
        remaining_rows = n_rows - row_idx - 1
        diffs_needed_future = remaining_rows * diffs_per_row
        
        total_needed = diffs_needed_current + diffs_needed_future
        available = M - len(used_diffs)
        
        if available < total_needed:
            return None, False
            
        # Determine next element
        if col_idx == 0:
            # Must be 0
            new_val = 0
            # Check if 0 is allowed (it is, it's the start)
            # But we must ensure 0 is not in used_diffs? No, 0 is not a difference.
            # Differences are positive.
            new_diffs = set()
            
            # Recurse
            current_row.append(new_val)
            res, success = dfs(row_idx, col_idx + 1, current_row, used_diffs, M, node_limit, nodes_visited)
            current_row.pop()
            
            if success:
                return res, True
            return None, False
        else:
            # Try next values
            # Lower bound: current_row[-1] + 1
            # Upper bound: M
            start_val = current_row[-1] + 1
            
            for x in range(start_val, M + 1):
                # Check differences
                new_diffs = set()
                valid = True
                for y in current_row:
                    d = x - y
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Try this x
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                res, success = dfs(row_idx, col_idx + 1, current_row, used_diffs, M, node_limit, nodes_visited)
                
                used_diffs.difference_update(new_diffs)
                current_row.pop()
                
                if success:
                    return res, True
                    
            return None, False

    # We need to store the rows to return them
    # Modifying dfs to return the rows list is tricky with backtracking.
    # Better: dfs returns True/False, and we maintain a global 'current_rows' list?
    # Or dfs returns the completed rows if successful.
    
    # Let's refactor dfs to build the rows list as it goes.
    
    def dfs_v2(row_idx, col_idx, current_row, used_diffs, M, node_limit, nodes_visited, rows):
        nodes_visited[0] += 1
        if nodes_visited[0] > node_limit:
            return False
        if time.time() - start_time > time_limit:
            return False
            
        if row_idx == n_rows:
            return True
            
        # Pruning
        diffs_in_current = col_idx * (col_idx - 1) // 2
        diffs_needed_current = diffs_per_row - diffs_in_current
        remaining_rows = n_rows - row_idx - 1
        diffs_needed_future = remaining_rows * diffs_per_row
        total_needed = diffs_needed_current + diffs_needed_future
        available = M - len(used_diffs)
        
        if available < total_needed:
            return False
            
        if col_idx == 0:
            rows.append([0])
            success = dfs_v2(row_idx, 1, [0], used_diffs, M, node_limit, nodes_visited, rows)
            if not success:
                rows.pop()
            return success
        else:
            start_val = current_row[-1] + 1
            for x in range(start_val, M + 1):
                new_diffs = set()
                valid = True
                for y in current_row:
                    d = x - y
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                success = dfs_v2(row_idx, col_idx + 1, current_row, used_diffs, M, node_limit, nodes_visited, rows)
                
                used_diffs.difference_update(new_diffs)
                current_row.pop()
                
                if success:
                    rows[-1] = list(current_row) # Store the completed row
                    return True
                    
            return False

    # Search loop
    for M in range(M_min, 121): # Try up to 120
        if time.time() - start_time > time_limit:
            break
            
        nodes_visited = [0]
        rows = []
        used_diffs = set()
        
        # We can add a small randomization or heuristic here if needed, 
        # but simple DFS with pruning is often enough for small M.
        # To avoid getting stuck in one bad branch, we could randomize the order of x?
        # No, smaller x is generally better for packing.
        
        success = dfs_v2(0, 0, [], used_diffs, M, 500000, nodes_visited, rows)
        
        if success:
            # Verify solution
            if is_valid_solution(rows):
                return rows
            else:
                # Should not happen if logic is correct
                pass
                
    # If no solution found within limit, return a valid but potentially suboptimal solution
    # Fallback to a greedy solution to ensure we return *something* valid
    # Or return the last best attempt?
    # The prompt says "Valid when...". If we return invalid, score is 0.
    # We must ensure we return a valid solution.
    # The greedy seed is valid. We can replicate it as fallback.
    
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
