import time
import random
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    best_rows = None
    best_max_val = float('inf')
    
    # Initial greedy solution to have a baseline
    # This helps in pruning if we find a solution quickly
    rows = greedy_construct(n_rows, k)
    if rows:
        max_val = max(max(row) for row in rows)
        best_rows = rows
        best_max_val = max_val

    # Backtracking search
    # We will try to improve upon the greedy solution
    # used_mask will be a Python integer where bit d is 1 if difference d is used
    used_mask = 0
    current_rows = []
    
    def backtrack(row_idx, current_row, max_val_so_far):
        nonlocal best_rows, best_max_val, used_mask
        
        # Time check
        if time.time() - start_time > time_limit:
            return False
            
        # Pruning: if current max is already worse than best, stop
        if max_val_so_far >= best_max_val:
            return False
            
        if row_idx == n_rows:
            # Found a valid solution
            if max_val_so_far < best_max_val:
                best_max_val = max_val_so_far
                best_rows = [list(r) for r in current_rows]
            return True
            
        # Build the next row
        # Start with 0
        row = [0]
        # We need to fill k more elements
        # We try to find the smallest valid x
        
        # To speed up, we can estimate the minimum possible next values
        # But simple greedy smallest-first is usually good.
        
        # We use a helper to find the next valid x
        def find_next_x(row, used_mask, start_x):
            x = start_x
            while True:
                # Check differences
                # new_diffs = {x - a for a in row}
                # mask = sum(1 << d for d in new_diffs)
                # if mask & used_mask == 0: return x
                # Optimization: compute mask incrementally or just check
                valid = True
                mask = 0
                for a in row:
                    d = x - a
                    if d > 0:
                        if (used_mask >> d) & 1:
                            valid = False
                            break
                        mask |= (1 << d)
                if valid:
                    return x, mask
                x += 1
                
        # Recursive helper for filling the current row
        def fill_row(row, pos, max_val_so_far):
            nonlocal used_mask
            if time.time() - start_time > time_limit:
                return False
                
            if pos == k + 1:
                # Row complete
                # Update used_mask
                # We need to apply the mask for this row
                # Actually, we should pass the mask for the row to the next level
                # Let's restructure: 
                # When we add an element, we update a local mask for the row.
                # When row is complete, we merge it into global used_mask.
                
                # Save state
                old_used_mask = used_mask
                # The mask for this row is already accumulated in 'row_mask' variable in closure?
                # Let's use a different approach: pass row_mask as argument.
                
                # Let's refactor to pass row_mask
                pass
            
            # This nested function structure is getting complicated with state.
            # Let's use a simpler recursive structure that returns the solution or None.
            return False
            
        # Let's implement a cleaner backtracking for a single row
        def solve_row(row_idx, prev_max, global_mask):
            nonlocal best_rows, best_max_val
            
            if time.time() - start_time > time_limit:
                return None
                
            if row_idx == n_rows:
                if prev_max < best_max_val:
                    best_max_val = prev_max
                    best_rows = [list(r) for r in current_rows]
                return True
                
            row = [0]
            row_mask = 0 # Mask of differences within this row
            
            def extend_row(pos, last_x, current_max):
                nonlocal row_mask
                if time.time() - start_time > time_limit:
                    return False
                
                if current_max >= best_max_val:
                    return False
                
                if pos == k + 1:
                    # Row complete
                    current_rows.append(row)
                    new_global_mask = global_mask | row_mask
                    if solve_row(row_idx + 1, current_max, new_global_mask):
                        return True
                    current_rows.pop()
                    return False
                
                # Find next x
                # Heuristic: try smallest x first
                x = last_x + 1
                # Optional: randomization for diversification
                # if random.random() < 0.1:
                #     x = last_x + 1 + random.randint(0, 5)
                
                while True:
                    # Check if x is valid
                    valid = True
                    new_diffs_mask = 0
                    for a in row:
                        d = x - a
                        if (global_mask >> d) & 1 or (row_mask >> d) & 1:
                            valid = False
                            break
                        new_diffs_mask |= (1 << d)
                    
                    if valid:
                        # Try this x
                        row.append(x)
                        old_row_mask = row_mask
                        row_mask |= new_diffs_mask
                        
                        if extend_row(pos + 1, x, x):
                            return True
                            
                        # Backtrack
                        row.pop()
                        row_mask = old_row_mask
                    
                    x += 1
                    
            return extend_row(1, 0, 0)
            
        solve_row(0, 0, 0)
        
    return best_rows

def greedy_construct(n_rows, k):
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            new_mask = 0
            for a in row:
                d = x - a
                if (used_mask >> d) & 1:
                    valid = False
                    break
                new_mask |= (1 << d)
            if valid:
                row.append(x)
                used_mask |= new_mask
        rows.append(row)
    return rows
