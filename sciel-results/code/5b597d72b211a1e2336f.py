import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    # Keep a safety margin. Evaluator kills at 150s.
    # We aim to finish well before that.
    deadline = start_time + 120.0 

    # Helper to calculate diffs of a row
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i):
                diffs.add(row[i] - row[j])
        return diffs

    # Simple Greedy Row Builder
    # Finds a row of length k+1 starting with 0, strictly increasing,
    # such that its diffs don't intersect 'forbidden'.
    # Tries to keep values small.
    def simple_greedy_row(forbidden, k):
        row = [0]
        used_local = set()
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden) and not (new_diffs & used_local):
                    row.append(x)
                    used_local |= new_diffs
                    break
                x += 1
        return row

    # DFS to find row with max <= limit
    # Returns None if not found within time_limit
    def find_row_dfs(forbidden, k, limit, time_limit):
        t_start = time.time()
        row = [0]
        local_diffs = set()
        
        # Heuristic: Try smaller values first
        # We need to choose a_i such that a_i > a_{i-1} and a_i <= limit
        # And diffs a_i - a_j are not in forbidden or local_diffs
        
        def backtrack(idx, last_val, current_row, current_diffs):
            if time.time() - t_start > time_limit:
                return None
            
            if idx == k + 1:
                return current_row[:]
            
            # Try candidates for current_row[idx]
            # Start from last_val + 1
            # End at limit
            
            # Optimization: If idx is the last one, we just need one valid x
            # We can iterate x from last_val + 1 to limit
            
            for x in range(last_val + 1, limit + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & forbidden:
                    continue
                if new_diffs & current_diffs:
                    continue
                
                current_row.append(x)
                current_diffs.update(new_diffs)
                
                res = backtrack(idx + 1, x, current_row, current_diffs)
                if res is not None:
                    return res
                
                current_row.pop()
                current_diffs.difference_update(new_diffs)
            
            return None

        return backtrack(1, 0, [0], set())

    # 1. Initial Construction
    used_diffs = set()
    rows = []
    for i in range(n_rows):
        # Use greedy first for speed and reliability
        row = simple_greedy_row(used_diffs, k)
        rows.append(row)
        used_diffs.update(get_diffs(row))
        
    current_max = max(max(r) for r in rows)
    
    # 2. Local Search / Optimization
    # We want to minimize current_max.
    # Strategy: Rebuild rows that are at current_max to see if we can get them lower.
    # If we can't, we might need to perturb other rows to free up space.
    
    # To avoid infinite loops or slow progress, we track the best solution found.
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # Time management
    # Phase 1: Aggressive reduction of max
    # Phase 2: Perturbation to escape local minima
    
    while time.time() < deadline:
        # Identify rows with the current global max
        critical_indices = [i for i, r in enumerate(rows) if max(r) == current_max]
        
        if not critical_indices:
            break # Should not happen, but safety
            
        # Pick a random critical row to rebuild
        idx = random.choice(critical_indices)
        old_row = rows[idx]
        
        # Remove its diffs
        old_diffs = get_diffs(old_row)
        used_diffs.difference_update(old_diffs)
        
        # Try to find a new row with max < current_max
        # Limit search to current_max - 1
        new_row = find_row_dfs(used_diffs, k, current_max - 1, time_limit=0.5)
        
        if new_row:
            # Success: we reduced the max for this row
            rows[idx] = new_row
            used_diffs.update(get_diffs(new_row))
            # Recalculate global max
            current_max = max(max(r) for r in rows)
            
            # Update best
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Failed to reduce. 
            # Option A: Revert and try a different critical row.
            # Option B: Rebuild with max == current_max but different structure (perturbation).
            # Let's try perturbation with some probability to escape local minima.
            if random.random() < 0.3:
                # Try to rebuild this same row with max == current_max
                # This might change which diffs are used, allowing other rows to shrink later.
                new_row = find_row_dfs(used_diffs, k, current_max, time_limit=0.5)
                if new_row:
                    # Ensure it's actually different or just accept it?
                    # If it's the same, it's a waste.
                    if new_row != old_row:
                        rows[idx] = new_row
                        used_diffs.update(get_diffs(new_row))
                        # Max hasn't changed, but state has.
                        # Recalculate max just in case (should be same)
                        current_max = max(max(r) for r in rows)
                    else:
                        # Revert
                        used_diffs.update(old_diffs)
                else:
                    # Revert
                    used_diffs.update(old_diffs)
            else:
                # Revert
                used_diffs.update(old_diffs)
                
                # If we are stuck, try rebuilding a NON-critical row
                # This might free up "small" differences that a critical row needs.
                non_critical = [i for i, r in enumerate(rows) if max(r) < current_max]
                if non_critical:
                    r_idx = random.choice(non_critical)
                    r_old = rows[r_idx]
                    r_old_diffs = get_diffs(r_old)
                    used_diffs.difference_update(r_old_diffs)
                    
                    # Rebuild with max <= current_max
                    r_new = find_row_dfs(used_diffs, k, current_max, time_limit=0.5)
                    if r_new:
                        rows[r_idx] = r_new
                        used_diffs.update(get_diffs(r_new))
                        # Max shouldn't increase
                        current_max = max(max(r) for r in rows)
                    else:
                        used_diffs.update(r_old_diffs)

    # Return the best solution found
    # The prompt asks for the result of construct().
    # We should return the valid solution with the smallest max.
    # best_rows is tracked.
    return best_rows
