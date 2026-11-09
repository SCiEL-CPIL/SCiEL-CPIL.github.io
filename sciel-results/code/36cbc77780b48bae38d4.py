import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    # We will search for a set of n_rows rows, each of length k+1,
    # starting at 0, with distinct differences, minimizing the max element.
    
    best_solution = None
    best_max = float('inf')
    
    # To speed up, we can use a global variable for the best max
    # and prune branches that cannot beat it.
    
    def dfs(current_rows, used_diffs, current_row, idx):
        nonlocal best_solution, best_max
        
        if len(current_rows) == n_rows:
            # Found a complete solution
            current_max = max(max(row) for row in current_rows)
            if current_max < best_max:
                best_max = current_max
                best_solution = [row[:] for row in current_rows]
            return
        
        # If we are building a row
        if idx < k:
            # Choose the next element for current_row
            # The last element in current_row is current_row[-1]
            # We try values from current_row[-1] + 1 upwards
            # Pruning: if current_row[-1] >= best_max, we can stop? 
            # Not exactly, we are building a row, the final element will be larger.
            # But if the current partial row's last element is already >= best_max, 
            # any extension will have a max >= best_max, so we can prune.
            if current_row and current_row[-1] >= best_max:
                return
                
            start_val = current_row[-1] + 1 if current_row else 1
            
            # To limit the search space, we can bound the max value we try.
            # The theoretical max is unbounded, but we can use best_max - 1 as a bound 
            # if we are not the first row? No, the max of the whole solution is the max of all rows.
            # If we are constructing the last row, the max of this row is the max of the solution.
            # If we are constructing an earlier row, the max of the solution is max(max(current_rows), max(new_row)).
            # So we can prune if max(current_row[-1], max(max(r) for r in current_rows)) >= best_max.
            
            # Let's try values from start_val to some limit.
            # A good limit is best_max - 1, because if we exceed best_max, we can't improve.
            # However, if best_max is inf, we need a reasonable upper bound.
            # We can use a greedy upper bound initially.
            
            # Let's generate a greedy solution first to set an initial best_max.
            
            for x in range(start_val, best_max):
                # Check if differences x - a for a in current_row are in used_diffs
                # Also, they must be distinct within the new differences (they are, since a is distinct and x is fixed)
                # But we need to check against used_diffs
                new_diffs = set()
                conflict = False
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Also, we must ensure that the new differences are distinct among themselves?
                # Yes, because a_i are distinct.
                
                # Recurse
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                dfs(current_rows, used_diffs, current_row, idx + 1)
                
                # Backtrack
                used_diffs.difference_update(new_diffs)
                current_row.pop()
                
        else:
            # Finished a row
            # The max of this row is current_row[-1]
            # Check if this row's max is less than best_max
            if current_row[-1] >= best_max:
                return
            
            # Add row to current_rows
            current_rows.append(current_row[:])
            
            # Recurse to next row
            dfs(current_rows, used_diffs, [0], 0)
            
            # Backtrack
            current_rows.pop()

    # Initial greedy solution to set best_max
    used = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        greedy_rows.append(row)
    
    best_max = max(max(r) for r in greedy_rows)
    best_solution = greedy_rows
    
    # Now run DFS to improve
    # We need to be careful with the recursion depth and branching factor.
    # For k=7, the branching factor can be large.
    # We can add a timeout or limit the search.
    
    # To make it faster, we can order the rows by the difficulty or something?
    # Or we can just run the DFS.
    
    # One optimization: when building a row, if the current partial row's last element 
    # plus the minimum possible increment for the remaining elements exceeds best_max, prune.
    # Min increment is 1 for each of the remaining (k - idx) elements.
    # So if current_row[-1] + (k - idx) >= best_max, prune.
    
    def dfs_opt(current_rows, used_diffs, current_row, idx):
        nonlocal best_solution, best_max
        
        if len(current_rows) == n_rows:
            current_max = max(max(row) for row in current_rows)
            if current_max < best_max:
                best_max = current_max
                best_solution = [row[:] for row in current_rows]
            return
        
        # Pruning: if we are building a row
        if idx < k:
            if current_row:
                last = current_row[-1]
                # Minimum possible final value for this row
                min_final = last + (k - idx)
                if min_final >= best_max:
                    return
            
            start_val = current_row[-1] + 1 if current_row else 1
            
            # We can also prune if the current row's last element is already >= best_max
            # (covered by min_final check)
            
            # Iterate x
            # To find smaller max, we iterate x in increasing order.
            # But we can stop if x >= best_max
            for x in range(start_val, best_max):
                new_diffs = set()
                conflict = False
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                dfs_opt(current_rows, used_diffs, current_row, idx + 1)
                
                used_diffs.difference_update(new_diffs)
                current_row.pop()
                
        else:
            # Finished a row
            if current_row[-1] >= best_max:
                return
            
            current_rows.append(current_row[:])
            
            dfs_opt(current_rows, used_diffs, [0], 0)
            
            current_rows.pop()

    # Run the optimized DFS
    # To avoid timeouts, we can run it for a limited number of nodes or time.
    # But 120s is generous for small n, k.
    # Let's add a simple node counter to stop if it takes too long.
    
    node_count = 0
    max_nodes = 1000000  # Limit to avoid timeout
    
    def dfs_limited(current_rows, used_diffs, current_row, idx):
        nonlocal best_solution, best_max, node_count
        
        if node_count > max_nodes:
            return
        
        node_count += 1
        
        if len(current_rows) == n_rows:
            current_max = max(max(row) for row in current_rows)
            if current_max < best_max:
                best_max = current_max
                best_solution = [row[:] for row in current_rows]
            return
        
        if idx < k:
            if current_row:
                last = current_row[-1]
                min_final = last + (k - idx)
                if min_final >= best_max:
                    return
            
            start_val = current_row[-1] + 1 if current_row else 1
            
            for x in range(start_val, best_max):
                new_diffs = set()
                conflict = False
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                dfs_limited(current_rows, used_diffs, current_row, idx + 1)
                
                used_diffs.difference_update(new_diffs)
                current_row.pop()
                
        else:
            if current_row[-1] >= best_max:
                return
            
            current_rows.append(current_row[:])
            
            dfs_limited(current_rows, used_diffs, [0], 0)
            
            current_rows.pop()

    dfs_limited([], set(), [0], 0)
    
    return best_solution
