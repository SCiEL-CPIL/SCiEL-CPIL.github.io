import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are globally distinct.
    Minimize the largest entry.
    
    Uses Binary Search on the scope (max entry) with Bitset-Accelerated
    Randomized Backtracking.
    """
    start_time = time.time()
    max_time = 115.0
    
    # Upper bound estimation
    # Counting bound: n_rows * k * (k+1) / 2
    min_possible_scope = n_rows * k * (k + 1) // 2
    # Add a margin
    upper_bound = min_possible_scope + 20
    lower_bound = min_possible_scope
    
    best_rows = None
    best_scope = float('inf')
    
    # We will iterate downwards from upper_bound.
    for scope in range(upper_bound, lower_bound - 1, -1):
        if time.time() - start_time > max_time:
            break
            
        found = False
        attempts = 30 # Increase attempts to ensure we find a solution if one exists
        
        for attempt in range(attempts):
            if time.time() - start_time > max_time:
                break
                
            rng = random.Random(attempt * 1000 + scope)
            
            # State for this attempt
            used_diffs = 0
            rows = []
            
            def backtrack(row_idx, current_row, used_diffs_mask, rows_list):
                nonlocal time, start_time, max_time
                
                if time.time() - start_time > max_time:
                    return False
                
                if row_idx == n_rows:
                    return True
                
                if len(current_row) == 0:
                    return backtrack(row_idx, [0], used_diffs_mask, rows_list)
                
                if len(current_row) == k + 1:
                    rows_list.append(current_row)
                    success = backtrack(row_idx + 1, [], used_diffs_mask, rows_list)
                    rows_list.pop()
                    return success
                
                last_val = current_row[-1]
                
                # Generate candidates
                # To improve performance, we can limit the candidate range.
                # But scope is small, so range(last_val+1, scope+1) is fine.
                
                candidates = list(range(last_val + 1, scope + 1))
                
                # Randomize order
                rng.shuffle(candidates)
                
                for cand in candidates:
                    new_diffs_mask = 0
                    valid = True
                    for a in current_row:
                        d = cand - a
                        if d > scope:
                            valid = False
                            break
                        mask = 1 << d
                        if used_diffs_mask & mask:
                            valid = False
                            break
                        new_diffs_mask |= mask
                    
                    if not valid:
                        continue
                    
                    # Recurse
                    current_row.append(cand)
                    
                    if backtrack(row_idx, current_row, used_diffs_mask | new_diffs_mask, rows_list):
                        return True
                    
                    # Backtrack
                    current_row.pop()
                
                return False
            
            if backtrack(0, [], 0, rows):
                found = True
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_rows = [r[:] for r in rows]
                break # Found a solution for this scope, try lower scope
        
        if not found:
            break

    if best_rows is None:
        # Fallback: Pure greedy
        used, rows = set(), []
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
    
    return best_rows
