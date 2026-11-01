import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Generate an initial upper bound using a greedy heuristic
    def greedy_construction():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return max(max(r) for r in rows)

    best_max = greedy_construction()
    
    # 2. Find a good first row (lexicographically smallest Golomb ruler of length k+1)
    # This is a small search, can be done quickly
    def find_best_first_row():
        # Backtrack to find the smallest max element for a single row
        # Then among those, lexicographically smallest
        # For k=7, this is very fast
        used = set()
        row = [0]
        
        def backtrack(pos, current_max):
            if time.time() > deadline:
                return None
            if len(row) == k + 1:
                return row[:]
            
            x = row[-1] + 1
            while True:
                if x > current_max:
                    return None
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used.update(new_diffs)
                    res = backtrack(len(row), max(current_max, x))
                    if res is not None:
                        return res
                    row.pop()
                    used.difference_update(new_diffs)
                x += 1
                
        # We need to find the minimal max first
        # We can just let the backtracking find the first solution, 
        # which will be the lexicographically smallest for a given max.
        # But we want the smallest max.
        # So we iterate max_val from lower bound upwards.
        
        # Lower bound for Golomb ruler of n marks is roughly n^2/4
        # For 8 marks, min max is around 25-30.
        # Let's just use the greedy result for the first row as a bound?
        # Or just run the backtracking without a fixed max, but it will find the 
        # lexicographically smallest sequence, which might not have the smallest max.
        # Actually, the lexicographically smallest Golomb ruler usually has a small max.
        
        # Let's just find the lexicographically smallest valid row.
        res = backtrack(0, float('inf'))
        return res

    first_row = find_best_first_row()
    if first_row is None:
        # Fallback to greedy
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    # 3. Backtrack for the remaining rows
    used_diffs = set()
    for i in range(1, len(first_row)):
        for j in range(i):
            used_diffs.add(first_row[i] - first_row[j])
            
    current_rows = [first_row]
    
    def solve(row_idx, pos, current_row):
        nonlocal best_max
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            if pos == k + 1:
                current_max = max(max(r) for r in current_rows)
                if current_max < best_max:
                    best_max = current_max
                    # Store best solution
                    return True
            return False
            
        if pos == k + 1:
            # Move to next row
            current_rows.append(current_row)
            res = solve(row_idx + 1, 0, [0])
            if res:
                return True
            current_rows.pop()
            return False
            
        # Pruning
        if current_row[-1] >= best_max:
            return False
            
        x = current_row[-1] + 1
        while x < best_max:
            if time.time() > deadline:
                return False
            new_diffs = {x - a for a in current_row}
            if not new_diffs & used_diffs:
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if solve(row_idx, pos + 1, current_row):
                    return True
                    
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            x += 1
            
        return False

    # Run the solver
    solve(1, 0, [0])
    
    # Reconstruct the best solution found
    # We need to store the actual rows, not just the max.
    # The above solver returns True/False but doesn't store the rows.
    # We need to modify it to store the best rows.
    
    # Let's rewrite the solver to store the best solution.
    best_solution = None
    
    def solve_store(row_idx, pos, current_row):
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            if pos == k + 1:
                current_max = max(max(r) for r in current_rows)
                if current_max < best_max:
                    best_max = current_max
                    best_solution = [r[:] for r in current_rows]
                    return True
            return False
            
        if pos == k + 1:
            current_rows.append(current_row)
            res = solve_store(row_idx + 1, 0, [0])
            if res:
                return True
            current_rows.pop()
            return False
            
        if current_row[-1] >= best_max:
            return False
            
        x = current_row[-1] + 1
        while x < best_max:
            if time.time() > deadline:
                return False
            new_diffs = {x - a for a in current_row}
            if not new_diffs & used_diffs:
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if solve_store(row_idx, pos + 1, current_row):
                    return True
                    
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            x += 1
            
        return False

    # Reset state for second run
    used_diffs = set()
    for i in range(1, len(first_row)):
        for j in range(i):
            used_diffs.add(first_row[i] - first_row[j])
    current_rows = [first_row]
    
    solve_store(1, 0, [0])
    
    if best_solution:
        return best_solution
    else:
        # Return greedy solution if backtracking failed or timed out
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows
