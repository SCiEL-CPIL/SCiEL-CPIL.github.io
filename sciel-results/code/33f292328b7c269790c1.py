import time
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety/overhead
    
    def greedy_construct():
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
        return rows

    # 1. Get initial greedy solution
    best_rows = greedy_construct()
    best_max = max(max(row) for row in best_rows)
    
    # If n_rows or k is very small, greedy might be optimal or close.
    # But we try to improve.
    
    # 2. Backtracking search to improve
    # We search for solutions with max value <= limit
    def search(limit):
        # Returns a list of rows if found, else None
        # Uses global deadline check
        if time.time() > deadline:
            return None
            
        used_diffs = set()
        current_rows = []
        
        def backtrack(row_idx, current_row, last_val):
            if time.time() > deadline:
                return False
                
            if row_idx == n_rows:
                # All rows constructed
                # We need to store the result
                # To avoid copying large lists, we can pass a reference or store in nonlocal
                nonlocal found_solution
                found_solution = [r[:] for r in current_rows]
                return True
                
            # If we are starting a new row
            if not current_row:
                current_row = [0]
                last_val = 0
            
            # If current row is full, move to next row
            if len(current_row) == k + 1:
                current_rows.append(current_row)
                # Recurse to next row
                res = backtrack(row_idx + 1, [], 0)
                if res:
                    return True
                # Backtrack
                current_rows.pop()
                return False
            
            # Try to add next element to current_row
            # Start from last_val + 1
            # Must be <= limit
            next_start = last_val + 1
            if next_start > limit:
                return False
                
            for x in range(next_start, limit + 1):
                # Check validity
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                    
                if not valid:
                    continue
                
                # Tentatively add x
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                # Recurse
                res = backtrack(row_idx, current_row, x)
                if res:
                    return True
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
                
            return False
        
        found_solution = None
        # Start with empty rows
        res = backtrack(0, [], 0)
        if res:
            return found_solution
        return None

    # Iterative improvement
    current_limit = best_max - 1
    
    # We want to find the smallest max.
    # If search(limit) fails, it means no solution with max <= limit.
    # Since we have a solution with max = limit + 1 (if we started from best_max), 
    # then limit + 1 is the optimal max.
    
    while current_limit >= 0:
        if time.time() > deadline:
            break
            
        # Try to find a solution with max <= current_limit
        sol = search(current_limit)
        
        if sol is not None:
            # Found a better solution
            new_max = max(max(row) for row in sol)
            if new_max < best_max:
                best_rows = sol
                best_max = new_max
                current_limit = new_max - 1
            else:
                # This shouldn't happen if search respects limit, 
                # unless limit was >= best_max which we don't do.
                break
        else:
            # No solution with max <= current_limit
            # Since we have a solution with max = current_limit + 1 (implied by previous step or initial),
            # we have found the optimal.
            break
            
    return best_rows
