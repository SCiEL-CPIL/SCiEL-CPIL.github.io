import time
import sys
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Parameters
    diffs_per_row = k * (k - 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound on scope
    lower_bound = total_diffs_needed
    
    # Helper to compute all pairwise differences in a row
    def row_diffs(row):
        return set(row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row)))
    
    # Helper to check if a set of differences conflicts with used
    def has_conflict(diffs, used):
        return bool(diffs & used)
    
    # Greedy construction to get an initial upper bound
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = set(x - a for a in row)
                if not has_conflict(new_diffs, used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, max(max(r) for r in rows)
    
    # Get initial greedy solution
    greedy_rows, greedy_scope = greedy_construct()
    
    # For (7,5), we know 112 works. Use that as a strong upper bound if greedy is worse.
    if n_rows == 7 and k == 5:
        target_upper = min(greedy_scope, 112)
    else:
        target_upper = greedy_scope
    
    # Local search: element-level moves
    def local_search(rows, scope_limit, max_iters=50000):
        used = set()
        for row in rows:
            used |= row_diffs(row)
        
        best_rows = [r[:] for r in rows]
        best_scope = max(max(r) for r in best_rows)
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
            
            # Randomly pick a row and an element to modify
            row_idx = random.randint(0, n_rows - 1)
            elem_idx = random.randint(1, k)  # Don't modify 0
            
            old_val = rows[row_idx][elem_idx]
            
            # Try to find a new value
            # We want to keep the row increasing: rows[row_idx][elem_idx-1] < new_val < rows[row_idx][elem_idx+1]
            min_val = rows[row_idx][elem_idx - 1] + 1
            max_val = scope_limit
            if elem_idx < k:
                max_val = min(max_val, rows[row_idx][elem_idx + 1] - 1)
            
            if min_val > max_val:
                continue
            
            # Try a few random candidates
            tried = set()
            found_new = False
            for _ in range(20):
                if min_val > max_val:
                    break
                new_val = random.randint(min_val, max_val)
                if new_val in tried or new_val == old_val:
                    continue
                tried.add(new_val)
                
                # Temporarily update the row
                old_row = rows[row_idx][:]
                rows[row_idx][elem_idx] = new_val
                
                # Check if the row is still strictly increasing
                if any(rows[row_idx][j] <= rows[row_idx][j-1] for j in range(1, len(rows[row_idx]))):
                    rows[row_idx] = old_row
                    continue
                
                # Compute new diffs for this row
                new_row_diffs = row_diffs(rows[row_idx])
                
                # Remove old diffs from used, add new ones, check for conflicts
                # More efficient: just check if new_row_diffs has any element in (used - old_row_diffs)
                old_row_diffs = row_diffs(old_row)
                
                # Check conflict: any diff in new_row_diffs that is in used but not in old_row_diffs
                conflict = False
                for d in new_row_diffs:
                    if d in used and d not in old_row_diffs:
                        conflict = True
                        break
                
                if conflict:
                    rows[row_idx] = old_row
                    continue
                
                # Valid move. Update used.
                used -= old_row_diffs
                used |= new_row_diffs
                
                # Check if we improved the scope
                current_scope = max(max(r) for r in rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                
                found_new = True
                break
            
            if not found_new:
                continue
        
        return best_rows, best_scope
    
    # Randomized restarts with local search
    best_rows = greedy_rows
    best_scope = greedy_scope
    
    # If greedy is already at the known optimal, return immediately
    if n_rows == 7 and k == 5 and greedy_scope <= 112:
        # Try to improve below 112
        pass
    
    # Try multiple random initializations + local search
    num_restarts = 0
    while time.time() - start_time < deadline - 10:
        num_restarts += 1
        
        # Create a random initial solution within scope_limit
        scope_limit = best_scope if best_scope < target_upper else target_upper
        
        # Try to build a random valid solution
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            # Randomly choose next elements
            for _ in range(k):
                # Choose a random x > row[-1]
                # To keep it within scope_limit, x <= scope_limit
                # Also need enough room for remaining elements
                remaining = k - len(row) + 1
                min_x = row[-1] + 1
                max_x = scope_limit
                if max_x < min_x:
                    success = False
                    break
                
                # Try a few random values
                found = False
                for _attempt in range(50):
                    x = random.randint(min_x, max_x)
                    new_diffs = set(x - a for a in row)
                    if not has_conflict(new_diffs, used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if not success:
                break
            
            rows.append(row)
        
        if not success or len(rows) != n_rows:
            # Try a different strategy: start from greedy and perturb
            if time.time() - start_time > deadline - 10:
                break
            
            # Perturb the greedy solution
            rows = [r[:] for r in greedy_rows]
            used = set()
            for row in rows:
                used |= row_diffs(row)
            
            # Do some random moves
            for _ in range(100):
                if time.time() - start_time > deadline - 10:
                    break
                row_idx = random.randint(0, n_rows - 1)
                elem_idx = random.randint(1, k)
                old_val = rows[row_idx][elem_idx]
                
                min_val = rows[row_idx][elem_idx - 1] + 1
                max_val = best_scope
                if elem_idx < k:
                    max_val = min(max_val, rows[row_idx][elem_idx + 1] - 1)
                
                if min_val > max_val:
                    continue
                
                new_val = random.randint(min_val, max_val)
                if new_val == old_val:
                    continue
                
                old_row = rows[row_idx][:]
                rows[row_idx][elem_idx] = new_val
                
                if any(rows[row_idx][j] <= rows[row_idx][j-1] for j in range(1, len(rows[row_idx]))):
                    rows[row_idx] = old_row
                    continue
                
                new_row_diffs = row_diffs(rows[row_idx])
                old_row_diffs = row_diffs(old_row)
                
                conflict = False
                for d in new_row_diffs:
                    if d in used and d not in old_row_diffs:
                        conflict = True
                        break
                
                if conflict:
                    rows[row_idx] = old_row
                    continue
                
                used -= old_row_diffs
                used |= new_row_diffs
        
        # Now do local search
        if len(rows) == n_rows:
            # Verify the solution is valid
            used_check = set()
            valid = True
            for row in rows:
                if row[0] != 0:
                    valid = False
                    break
                if any(row[j] <= row[j-1] for j in range(1, len(row))):
                    valid = False
                    break
                rd = row_diffs(row)
                if rd & used_check:
                    valid = False
                    break
                used_check |= rd
            
            if valid:
                scope = max(max(r) for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
                
                # Do local search to improve
                improved_rows, improved_scope = local_search(rows, scope, max_iters=2000)
                if improved_scope < best_scope:
                    best_scope = improved_scope
                    best_rows = improved_rows
        
        # Check if we've hit the lower bound
        if best_scope <= lower_bound:
            break
    
    return best_rows
