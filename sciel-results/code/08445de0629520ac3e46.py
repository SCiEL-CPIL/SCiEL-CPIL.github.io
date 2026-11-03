import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit
    
    used_diffs = set()
    rows = []
    
    # We will construct rows one by one.
    # To improve the result, we can try to construct rows in a specific order or use backtracking.
    # Given the small size (7 rows, 6 elements), backtracking per row is feasible.
    
    def backtrack_row(current_row, next_val, used_diffs):
        if len(current_row) == k + 1:
            return current_row[:]
        
        # Check time
        if time.time() > deadline:
            return None
            
        # Try values from next_val upwards
        # We can bound the search if we have a target scope, but we don't know it yet.
        # However, we can stop if next_val exceeds a reasonable bound.
        # The counting bound is 105. Best known 112. Let's say we don't want to exceed 200.
        # But actually, we just want the smallest possible.
        
        x = next_val
        while x < 500: # Arbitrary high bound, should be reached quickly if valid
            # Check if x creates conflicts
            new_diffs = {x - a for a in current_row}
            if not new_diffs & used_diffs:
                # Recurse
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                res = backtrack_row(current_row, x + 1, used_diffs)
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
                
                if res is not None:
                    return res
            x += 1
            
        return None

    for _ in range(n_rows):
        # Try to construct the next row
        # We start with [0]
        row = [0]
        # The next value must be > 0
        # We call backtrack_row which expects current_row and next_val
        # Note: backtrack_row checks len == k+1. Since row=[0], len=1.
        # It will try to add elements until len == k+1.
        
        # Optimization: We can pass a "hint" or just let it search.
        # To make it faster, we can pre-check if 0 is already used? No, 0 is not a difference.
        
        res_row = backtrack_row(row, 1, used_diffs)
        
        if res_row is None:
            # If backtracking fails (e.g., timeout or no solution), fallback to greedy for this row
            # Or just return what we have? The task requires valid output.
            # If backtracking fails due to timeout, we should use a greedy fallback to ensure completion.
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs.update(new_diffs)
                x += 1
            res_row = row
            
        rows.append(res_row)
        
        if time.time() > deadline:
            # If we run out of time, we must return a valid set.
            # We have constructed some rows. We need to fill the rest quickly.
            for _ in range(len(rows), n_rows):
                row = [0]
                x = 1
                while len(row) < k + 1:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs.update(new_diffs)
                    x += 1
                rows.append(row)
            break

    return rows
