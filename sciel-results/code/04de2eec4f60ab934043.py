def construct(n_rows, k):
    import time
    import sys
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    TIME_LIMIT = 115
    
    # Total differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound for M
    # Counting bound: M >= total_diffs_needed
    # Actually, differences are from 1 to M. So M >= total_diffs_needed.
    min_M = total_diffs_needed
    
    # Upper bound
    max_M = 200 # Safe upper bound for small n_rows, k
    
    used = set()
    rows = []
    
    def find_row(used_set, M):
        """Find a valid row with max element <= M, using only unused differences."""
        row = [0]
        
        def dfs(current_row, current_diffs):
            if time.time() - start_time > TIME_LIMIT:
                return None
            if len(current_row) == k + 1:
                return current_row[:]
            
            last = current_row[-1]
            remaining_slots = k + 1 - len(current_row)
            
            # Pruning: Can we fit remaining slots in [last+1, M]?
            if M - last < remaining_slots:
                return None
            
            # Pruning: Do we have enough unused differences in [1, M] to complete the row?
            # This is a weak bound, but we can check if (M - last) is enough.
            # Better: count available diffs? Too slow.
            
            # Heuristic: Try values that are "sparse" or "dense"?
            # Standard: try from last+1 upwards.
            # To improve, we can try values that create smaller differences first? 
            # Or larger? 
            # Smaller differences are more "scarce" (more likely to be used by others? No, we are constructing one row at a time).
            # Within one row, smaller differences are more constrained because they appear more often in the difference set?
            # Actually, let's just iterate.
            
            for next_val in range(last + 1, M + 1):
                # Check new differences
                valid = True
                new_diffs = []
                for a in current_row:
                    d = next_val - a
                    if d in used_set or d in current_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Recursive call
                # Pass a new set to avoid mutation overhead in deep recursion? 
                # Or mutate and undo. Mutation is faster.
                current_diffs.update(new_diffs)
                current_row.append(next_val)
                
                result = dfs(current_row, current_diffs)
                
                if result is not None:
                    return result
                
                # Undo
                current_row.pop()
                current_diffs.difference_update(new_diffs)
            
            return None
        
        # Start DFS
        # We can try different starting points?
        # The first element (after 0) can be anything from 1 to M.
        # To find *a* row, we can just start DFS from [0].
        # But DFS from [0] will try 1, 2, ...
        # This might be slow if 1 is a bad choice.
        # We can add a heuristic: try values that are "central" or "extreme"?
        # Let's stick to standard DFS for now, but optimize the inner loop.
        
        return dfs([0], set())
    
    def can_pack(M):
        """Check if we can pack n_rows rows with max element <= M."""
        nonlocal used, rows
        used = set()
        rows = []
        
        def backtrack(idx):
            if time.time() - start_time > TIME_LIMIT:
                return False
            if idx == n_rows:
                return True
            
            # Find a valid row
            # To improve, we can try to find a row that is "harder" or "easier"?
            # Since rows are symmetric, we just find any valid row.
            row = find_row(used, M)
            
            if row is None:
                return False
            
            # Mark differences as used
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    row_diffs.add(row[j] - row[i])
            
            used.update(row_diffs)
            rows.append(row)
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            rows.pop()
            used.difference_update(row_diffs)
            
            return False
        
        return backtrack(0)
    
    # Iterate M from min_M upwards
    # We can jump M if we want, but linear is safer for small ranges.
    for M in range(min_M, max_M + 1):
        if time.time() - start_time > TIME_LIMIT:
            break
        if can_pack(M):
            return rows
    
    # Fallback: greedy construction without strict scope limit
    # This ensures we return *something* valid, even if not optimal.
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used.update(new_diffs)
            x += 1
        rows.append(row)
    return rows
