import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Greedy Initial Solution
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            for a in row:
                if x - a in used:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]: # a is the new x? No, x is the new one.
                    used.add(x - a)
        rows.append(row)
    
    current_max = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # Helper to remove row diffs from used
    def remove_row_diffs(row, used):
        for i in range(len(row)):
            for j in range(i):
                used.discard(row[i] - row[j])
                
    # Helper to add row diffs to used
    def add_row_diffs(row, used):
        for i in range(len(row)):
            for j in range(i):
                used.add(row[i] - row[j])
                
    # DFS to reconstruct a single row with minimal max value
    def reconstruct_row(row_idx, used, deadline):
        # row_idx is the index in 'rows'
        # We want to find a new row that is valid with current 'used'
        # and has a small max value.
        
        # We know the row must start with 0.
        # We will search for the sequence.
        
        # To speed up, we can limit the max value we try.
        # We know the current max. Let's try to beat it.
        # Or just find any valid one, preferring small ones.
        
        # We'll use a recursive search.
        # State: current_row, last_val
        # To avoid deep recursion overhead, we can use a stack or just recursion (depth k is small, max 6).
        
        found = False
        best_local_row = None
        best_local_max = float('inf')
        
        # We can use a simple DFS that tries candidates in increasing order.
        # If it finds a solution, it's likely near-optimal for that branch.
        # To escape local minima, we might want to randomize candidate order occasionally.
        
        def dfs_reconstruct(current_row, used, deadline):
            nonlocal found, best_local_row, best_local_max
            
            if time.time() > deadline:
                return False
                
            if len(current_row) == k + 1:
                found = True
                curr_max = current_row[-1]
                if curr_max < best_local_max:
                    best_local_max = curr_max
                    best_local_row = current_row[:]
                # We can continue to see if we find an even smaller one?
                # Since we iterate x in increasing order, the first solution found is the smallest for this path.
                # But different paths might yield different maxes.
                # For simplicity, let's just return the first one found (smallest x at each step).
                # This tends to minimize the max.
                return True
                
            last_val = current_row[-1]
            # Try candidates
            # We can limit the search range. 
            # If last_val is already >= best_local_max, we can prune? 
            # Actually, if we are trying to minimize the max, and we already have a solution with max M,
            # we don't need to explore branches where the current last_val > M? 
            # Not exactly, because future values will be larger.
            # But if last_val > best_local_max, any completion will have max >= last_val > best_local_max.
            # So we can prune.
            
            if best_local_row is not None and last_val >= best_local_max:
                return False
                
            # Generate candidates
            # Start from last_val + 1
            # We can add a randomization factor to escape local optima
            # E.g., with prob 0.1, start from last_val + 2 or skip some
            
            start_x = last_val + 1
            if random.random() < 0.1:
                start_x += random.randint(0, 2)
                
            # Limit the number of candidates to try to avoid infinite loop in worst case
            # But k is small, so depth is small.
            for x in range(start_x, start_x + 100): # Arbitrary limit
                valid = True
                diffs = []
                for a in current_row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                    diffs.append(d)
                    
                if not valid:
                    continue
                    
                # Add diffs
                for d in diffs:
                    used.add(d)
                current_row.append(x)
                
                if dfs_reconstruct(current_row, used, deadline):
                    # Success found deeper
                    # We keep the state? No, we need to backtrack to return to caller
                    # But we stored best_local_row.
                    # We should remove diffs before returning?
                    # The caller expects 'used' to be restored if it backtracks.
                    # But if we found a solution, we might want to keep it?
                    # No, the caller will handle the final update.
                    # We just need to signal success.
                    # But we must clean up 'used' before returning to the loop in dfs_reconstruct?
                    # Yes.
                    for d in diffs:
                        used.remove(d)
                    current_row.pop()
                    return True
                    
                # Backtrack
                for d in diffs:
                    used.remove(d)
                current_row.pop()
                
            return False
            
        # Start DFS
        initial_row = [0]
        # Note: 0 has no diffs to add to 'used' yet? 
        # The diffs are between pairs. 0 is just the start.
        # So we don't add any diffs for 0.
        
        if dfs_reconstruct(initial_row, used, deadline):
            return best_local_row
        return None

    # Optimization Loop
    while time.time() < deadline:
        # Pick a random row to re-optimize
        idx = random.randint(0, n_rows - 1)
        
        # Remove current row's diffs
        old_row = rows[idx]
        remove_row_diffs(old_row, used)
        
        # Try to reconstruct
        new_row = reconstruct_row(idx, used, deadline)
        
        if new_row is not None:
            # Check if it's better
            if new_row[-1] < old_row[-1]:
                # Accept
                add_row_diffs(new_row, used)
                rows[idx] = new_row
                current_max = max(max(r) for r in rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
            else:
                # Restore old row
                add_row_diffs(old_row, used)
        else:
            # Failed to find a valid row, restore old
            add_row_diffs(old_row, used)
            
    return best_rows
