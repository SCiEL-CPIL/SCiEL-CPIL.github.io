"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 110.0
    
    # Calculate bounds
    total_diffs = n_rows * k * (k + 1) // 2
    low = total_diffs
    
    # Get an upper bound using greedy
    def greedy_upper_bound():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return max(max(r) for r in rows)
    
    high = greedy_upper_bound()
    
    # If the greedy bound is already low, we might just return it if we can't do better quickly.
    # But we want to minimize.
    
    def solve_for_M(M, seed):
        random.seed(seed)
        used_diffs = set()
        rows = []
        
        def dfs(row_idx, current_row):
            if time.time() - start_time > time_limit - 2:
                return False
            
            if row_idx == n_rows:
                return True
                
            if len(current_row) == k + 1:
                rows.append(current_row[:])
                res = dfs(row_idx + 1, [0])
                if res:
                    return True
                rows.pop()
                return False
                
            last_val = current_row[-1]
            
            # Generate candidates
            candidates = list(range(last_val + 1, M + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if dfs(row_idx, current_row):
                    return True
                    
                used_diffs.difference_update(new_diffs)
                current_row.pop()
                
            return False
            
        return dfs(0, [0])

    best_rows = None
    best_M = high
    
    # Try to find a solution for M from low to high
    # We can try a few random seeds for each M to increase robustness
    for M in range(low, high + 1):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Try a few random restarts for this M
        found = False
        for attempt in range(10):
            if time.time() - start_time > time_limit - 5:
                break
            if solve_for_M(M, start_time * 1000 + M * 100 + attempt):
                found = True
                break
                
        if found:
            best_M = M
            best_rows = [r[:] for r in rows]
            break
            
    if best_rows is None:
        # Fallback to greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        
    return best_rows
# EVOLVE-BLOCK-END
