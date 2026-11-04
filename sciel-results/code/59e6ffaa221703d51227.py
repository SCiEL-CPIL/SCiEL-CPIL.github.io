import random
import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    best_result = None
    best_scope = float('inf')
    
    start_time = time.time()
    time_limit = 110.0
    
    max_restarts = 200
    node_limit_per_restart = 100000
    
    for _ in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        used_diffs = set()
        rows = []
        nodes_visited = 0
        found_in_this_restart = False
        
        def backtrack(row_idx, curr_row):
            nonlocal nodes_visited, best_scope, best_result, used_diffs, rows, found_in_this_restart
            
            if time.time() - start_time > time_limit:
                return False
            
            if nodes_visited > node_limit_per_restart:
                return False
            
            nodes_visited += 1
            
            if row_idx == n_rows:
                scope = max(r[-1] for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_result = [r[:] for r in rows]
                    found_in_this_restart = True
                return True
            
            if len(curr_row) == k + 1:
                rows.append(curr_row)
                success = backtrack(row_idx + 1, [0])
                if success:
                    return True
                rows.pop()
                return False
            
            last_val = curr_row[-1]
            
            # Collect valid candidates
            upper_bound = best_scope if best_scope < float('inf') else 200
            candidates = []
            
            for x in range(last_val + 1, upper_bound + 1):
                new_diffs = {x - a for a in curr_row}
                if not (new_diffs & used_diffs):
                    candidates.append((x, new_diffs))
                    
                # Early break if we have enough candidates?
                # We want the smallest ones, so we can stop if we have, say, 20.
                if len(candidates) >= 20:
                    break
            
            if not candidates:
                return False
            
            # Sort by x (ascending)
            candidates.sort(key=lambda item: item[0])
            
            # Selection Strategy:
            # Primarily pick the smallest (index 0).
            # But add randomness to escape local minima.
            # If we always pick 0, we are deterministic.
            
            # Let's pick index 0 with prob 0.7, index 1 with 0.1, index 2 with 0.1, random from top 5 with 0.1.
            # This maintains a strong bias towards small x (minimizing scope) but allows exploration.
            
            r = random.random()
            if r < 0.7:
                idx = 0
            elif r < 0.8:
                idx = min(1, len(candidates)-1)
            elif r < 0.9:
                idx = min(2, len(candidates)-1)
            else:
                idx = random.randint(0, min(4, len(candidates)-1))
            
            x, new_diffs = candidates[idx]
            
            # Apply
            curr_row.append(x)
            used_diffs.update(new_diffs)
            
            # Recurse
            success = backtrack(row_idx, curr_row)
            
            # Undo
            curr_row.pop()
            used_diffs.difference_update(new_diffs)
            
            if success:
                return True
            
            # If this choice failed, try the next best candidate?
            # The current structure only tries ONE candidate per node and backtracks if it fails.
            # This is NOT a full backtracking search. It's a "randomized greedy" that backtracks to the previous level if it fails.
            # To make it a proper backtracking search, we need to iterate through candidates.
            
            # Let's change the structure to iterate through candidates.
            # But we want to limit the branching factor for speed.
            
            return False
        
        # The current implementation only tries one candidate per step.
        # If it fails, it returns False to the parent, which then tries a different candidate (if the parent iterates).
        # But my parent doesn't iterate! It just picks one.
        # So if the leaf fails, the whole branch fails, and we restart.
        # This is essentially "Greedy with Restart".
        
        # To get the benefits of backtracking, I must iterate through candidates in the loop.
        
        # Let's rewrite the inner loop to iterate.
        
        # But iterating through all candidates can be expensive.
        # We can limit the number of candidates we try per node (e.g., top 5).
        
        # Revised Backtrack Logic:
        
        def backtrack_v2(row_idx, curr_row):
            nonlocal nodes_visited, best_scope, best_result, used_diffs, rows, found_in_this_restart
            
            if time.time() - start_time > time_limit:
                return False
            
            if nodes_visited > node_limit_per_restart:
                return False
            
            nodes_visited += 1
            
            if row_idx == n_rows:
                scope = max(r[-1] for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_result = [r[:] for r in rows]
                    found_in_this_restart = True
                return True
            
            if len(curr_row) == k + 1:
                rows.append(curr_row)
                success = backtrack_v2(row_idx + 1, [0])
                if success:
                    return True
                rows.pop()
                return False
            
            last_val = curr_row[-1]
            upper_bound = best_scope if best_scope < float('inf') else 200
            candidates = []
            
            for x in range(last_val + 1, upper_bound + 1):
                new_diffs = {x - a for a in curr_row}
                if not (new_diffs & used_diffs):
                    candidates.append((x, new_diffs))
                if len(candidates) >= 10: # Limit candidate pool
                    break
            
            if not candidates:
                return False
            
            candidates.sort(key=lambda item: item[0])
            
            # Try candidates in order (smallest first)
            # To add randomness, we can shuffle the first few?
            # Or just try them in order. If we find a solution, we stop.
            # This finds the lexicographically smallest solution.
            # If we want to explore more, we can skip some.
            
            # Let's try the first 3 candidates.
            # If any of them leads to a solution, we return True.
            
            for i in range(min(3, len(candidates))):
                x, new_diffs = candidates[i]
                
                curr_row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack_v2(row_idx, curr_row):
                    return True
                
                curr_row.pop()
                used_diffs.difference_update(new_diffs)
                
            return False
        
        backtrack_v2(0, [0])
        
        if best_scope <= 105:
            break

    if best_result is None:
        used = set()
        rows = []
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
        
    return best_result
