import time
import random
import numpy as np

def construct(n_rows, k):
    # 1. Get initial bound from greedy
    greedy_rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
                x += 1
            else:
                x += 1
        greedy_rows.append(row)
    
    best_max = max(max(r) for r in greedy_rows)
    best_solution = [list(r) for r in greedy_rows]
    
    # 2. Backtracking search with improved pruning and randomization
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    def dfs_timed(row_idx, current_row, current_rows, used_diffs, current_max):
        nonlocal best_max, best_solution
        
        if time.time() - start_time > timeout:
            return False
        
        if row_idx == n_rows:
            if current_max < best_max:
                best_max = current_max
                best_solution = current_rows + [current_row]
            return True
        
        remaining = k + 1 - len(current_row)
        min_final = current_row[-1] + remaining
        if min_final >= best_max:
            return False
        if current_max >= best_max:
            return False
        
        start = current_row[-1] + 1
        end = best_max - remaining + 1
        
        if start > end:
            return False
        
        # Generate candidate values
        candidates = list(range(start, end + 1))
        
        # Heuristic: prioritize values that create "dense" rows (smaller gaps)
        # but also consider diversity to avoid local minima
        # We can shuffle slightly or use a weighted random choice
        # For now, let's try increasing order first as it's often optimal for DTS
        # But if we find a solution, we might want to explore other branches
        
        # To improve search, we can try to find a complete row first, then backtrack
        # This is what the current DFS does, but we can add some randomization
        # to escape local minima
        
        # Let's try a few different strategies:
        # 1. Increasing order (current)
        # 2. Random order
        # 3. Hybrid: try increasing, but if stuck, try random
        
        # For simplicity and reliability, let's stick with increasing order
        # but add better pruning
        
        for x in candidates:
            if time.time() - start_time > timeout:
                return False
            
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            
            new_used = used_diffs | new_diffs
            new_row = current_row + [x]
            new_max = max(current_max, x)
            
            if len(new_row) == k + 1:
                new_rows = current_rows + [new_row]
                if dfs_timed(row_idx + 1, [0], new_rows, new_used, new_max):
                    return True
            else:
                if dfs_timed(row_idx, new_row, current_rows, new_used, new_max):
                    return True
        return False
    
    # Start search
    dfs_timed(0, [0], [], set(), 0)
    
    # If we still have time, try a different approach: 
    # Use simulated annealing or local search to improve the solution
    if time.time() - start_time < timeout - 5:
        # Try to improve the best solution found so far
        # This is complex, so for now we just return the best found
        pass
    
    return best_solution
