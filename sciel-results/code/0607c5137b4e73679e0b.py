import time
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for cleanup/return
    
    # 1. Get a baseline solution using the greedy method (fast)
    def greedy_construct():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            for _ in range(k):
                x = last + 1
                while True:
                    # Calculate new differences
                    mask = 0
                    valid = True
                    for a in row:
                        d = x - a
                        bit = 1 << d
                        if used & bit:
                            valid = False
                            break
                        mask |= bit
                    if valid:
                        row.append(x)
                        last = x
                        used |= mask
                        break
                    x += 1
            rows.append(row)
        return rows, max(max(r) for r in rows)

    best_rows, best_scope = greedy_construct()
    
    # If n_rows or k is very small, greedy might be optimal or close enough
    # But let's try to improve with backtracking
    
    # State for backtracking
    # used_mask: int
    # current_rows: list of lists
    # current_row: list
    # row_idx: int
    # pos_in_row: int
    # current_max: int
    
    # We want to find a solution with scope < best_scope
    # We will prune if current_max >= best_scope
    
    # To make backtracking effective, we need a good heuristic.
    # Trying the smallest valid x first is standard.
    
    def backtrack(row_idx, pos_in_row, current_row, used_mask, current_max):
        nonlocal best_rows, best_scope, deadline
        
        if time.time() > deadline:
            return False  # Timeout signal
        
        # Pruning: if current max is already >= best_scope, no improvement possible
        if current_max >= best_scope:
            return False
        
        # If we have finished all rows
        if row_idx == n_rows:
            # Check if we actually filled the last row completely
            # The loop structure below handles filling rows.
            # If we exit the loop, we have n_rows complete rows.
            # Update best
            # We need to copy the rows
            new_rows = [r[:] for r in current_rows]
            # The last row is current_row
            # Wait, the structure needs to be consistent.
            # Let's say current_rows holds completed rows.
            # current_row holds the one being built.
            # When row_idx == n_rows, it means we just finished the last row?
            # No, row_idx is the index of the row being built.
            # If row_idx == n_rows, we are done.
            # But we need to ensure the last row was added to current_rows.
            
            # Let's restructure:
            # current_rows: list of completed rows
            # current_row: list of elements in the current row
            # row_idx: index of current row (0 to n_rows-1)
            # pos_in_row: index of next element to add (1 to k, since 0 is there)
            
            # If pos_in_row == k+1, the row is complete.
            # Add to current_rows, start next row.
            
            pass # Placeholder
        
        # If current row is complete
        if pos_in_row == k + 1:
            # Add current row to completed rows
            current_rows.append(current_row[:])
            current_row = [0]
            pos_in_row = 1
            row_idx += 1
            
            if row_idx == n_rows:
                # All rows complete
                # The last row was just added to current_rows
                # Calculate scope
                scope = max(max(r) for r in current_rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in current_rows]
                # We can continue searching for even smaller scope?
                # Yes, because we pruned based on old best_scope.
                # But if we found a smaller one, we can prune more aggressively now.
                # However, since we are in DFS, we will backtrack and try other branches.
                return True # Found a solution (better than previous best)
            
            # Start next row
            # The max of the new row will be at least 0, but we track global max.
            # current_max remains the same (it's the max of all elements so far)
            return backtrack(row_idx, pos_in_row, current_row, used_mask, current_max)
        
        # Try to add the next element to current_row
        # The next element must be > current_row[-1]
        # Let's iterate x from current_row[-1] + 1 upwards
        # But we can bound x.
        # x cannot be such that x - 0 >= best_scope (since 0 is in row, diff is x)
        # So x < best_scope
        # Also x - current_row[-1] is the smallest new diff? No, x - current_row[0] is the largest.
        
        last_val = current_row[-1]
        # We need to find x such that all diffs x-a are unused.
        
        # Optimization:
        # The differences are x - a for a in current_row.
        # Let's try x starting from last_val + 1.
        
        # To avoid infinite loop or excessive search, we can limit x.
        # If x >= best_scope, we prune (handled by current_max check in next step? 
        # No, current_max is updated after we pick x.
        # So if x >= best_scope, we can skip immediately.
        
        # Let's iterate x
        x = last_val + 1
        while x < best_scope:
            # Check if x is valid
            mask = 0
            valid = True
            for a in current_row:
                d = x - a
                bit = 1 << d
                if used_mask & bit:
                    valid = False
                    break
                mask |= bit
            
            if valid:
                # Tentatively add x
                new_current_row = current_row + [x]
                new_used_mask = used_mask | mask
                new_current_max = max(current_max, x)
                
                # Recurse
                # We need to pass a copy of current_row? 
                # Or modify and revert.
                # Modifying is faster.
                current_row.append(x)
                
                found = backtrack(row_idx, pos_in_row + 1, current_row, new_used_mask, new_current_max)
                
                current_row.pop()
                
                if found:
                    # We found a solution. 
                    # Should we continue searching for a better one?
                    # Yes, because we might find a smaller scope.
                    # But the search space is huge.
                    # If we found one, we updated best_scope.
                    # The pruning `if current_max >= best_scope` will now be tighter.
                    # So continuing is good.
                    # However, to save time, we could return True immediately if we think we are close to optimal.
                    # But we don't know.
                    # Let's continue to explore other branches.
                    pass
            
            x += 1
            
        return False

    # Initialize state
    current_rows = []
    current_row = [0]
    
    # Call backtrack
    # We catch the timeout implicitly by checking time in backtrack.
    # But if backtrack returns due to timeout, we just use best_rows.
    
    # One issue: The recursive call modifies `current_rows` and `current_row`.
    # We need to be careful with state management.
    # The `current_rows.append` inside `backtrack` modifies the list.
    # We need to revert it when backtracking.
    
    # Let's rewrite `backtrack` to handle state restoration properly.
    
    def backtrack_safe(row_idx, pos_in_row, current_row, used_mask, current_max):
        nonlocal best_rows, best_scope, deadline, current_rows
        
        if time.time() > deadline:
            return False
        
        if current_max >= best_scope:
            return False
        
        if pos_in_row == k + 1:
            # Row complete
            current_rows.append(current_row[:])
            
            if row_idx == n_rows - 1:
                # Last row
                scope = max(max(r) for r in current_rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in current_rows]
                # Revert append
                current_rows.pop()
                return True
            else:
                # Start next row
                new_row = [0]
                # Recurse
                found = backtrack_safe(row_idx + 1, 1, new_row, used_mask, current_max)
                # Revert append
                current_rows.pop()
                return found
        
        # Try next element
        last_val = current_row[-1]
        x = last_val + 1
        
        while x < best_scope:
            mask = 0
            valid = True
            for a in current_row:
                d = x - a
                bit = 1 << d
                if used_mask & bit:
                    valid = False
                    break
                mask |= bit
            
            if valid:
                current_row.append(x)
                new_max = max(current_max, x)
                
                found = backtrack_safe(row_idx, pos_in_row + 1, current_row, used_mask | mask, new_max)
                
                current_row.pop()
                
                if found:
                    # Continue searching for better solutions
                    # If we want to stop early, we could return True here.
                    # But continuing allows finding smaller scopes.
                    pass
            
            x += 1
            
        return False

    # Run the search
    # To avoid deep recursion overhead or stack overflow (though depth is small),
    # we can just call it.
    # Note: `current_rows` is a nonlocal variable.
    # We must ensure it's empty before start.
    current_rows = []
    backtrack_safe(0, 1, [0], 0, 0)
    
    return best_rows
