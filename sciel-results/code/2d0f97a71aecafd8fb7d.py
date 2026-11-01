import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    
    def time_left():
        return deadline - time.time()

    # Lower bound on M:
    # Total diffs = n_rows * diffs_per_row
    # Max diff = M
    # Number of possible diffs = M
    # So M >= n_rows * diffs_per_row
    lower_bound = n_rows * diffs_per_row
    
    # Upper bound from a simple greedy construction
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                found = False
                # Try small increments
                for x in range(last + 1, last + 1000):
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(x)
                        for d in new_diffs:
                            used_diffs.add(d)
                        found = True
                        break
                if not found:
                    # Fallback: just pick a large number
                    x = last + 1000
                    while True:
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            row.append(x)
                            for a in row[:-1]:
                                used_diffs.add(x - a)
                            break
                        x += 1
            rows.append(row)
        return rows

    # Get a good initial upper bound
    init_rows = greedy_init()
    current_best_M = max(max(r) for r in init_rows)
    
    # We will try to find a solution with M < current_best_M
    # We search for M in descending order from current_best_M - 1 down to lower_bound
    
    best_rows = init_rows
    
    # Backtracking search for a given M
    def solve_with_limit(M, limit):
        used_diffs = set()
        rows = [[0] * row_len for _ in range(n_rows)]
        nodes = [0]
        
        def dfs(r_idx, c_idx, min_last_for_next):
            if time_left() < 1:
                return False
            if nodes[0] > limit:
                return False
            
            nodes[0] += 1
            
            if r_idx == n_rows:
                return True
            
            if c_idx == row_len:
                # Row finished
                # Check symmetry breaking
                if rows[r_idx][-1] < min_last_for_next:
                    return False
                
                # Recurse to next row
                return dfs(r_idx + 1, 0, rows[r_idx][-1])
            
            if c_idx == 0:
                # Must be 0
                rows[r_idx][0] = 0
                return dfs(r_idx, c_idx + 1, min_last_for_next)
            else:
                # Previous element
                prev_val = rows[r_idx][c_idx - 1]
                
                # Lower bound for this value
                lb = prev_val + 1
                
                # Upper bound for this value
                # 1. Must be <= M
                # 2. If this is the last element (c_idx == row_len - 1), it must be >= min_last_for_next
                #    So if min_last_for_next > lb, lb = min_last_for_next
                if c_idx == row_len - 1:
                    if min_last_for_next > lb:
                        lb = min_last_for_next
                
                ub = M
                
                # Try values from lb to ub
                for val in range(lb, ub + 1):
                    # Check validity
                    valid = True
                    new_diffs = []
                    for i in range(c_idx):
                        d = val - rows[r_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Commit
                    rows[r_idx][c_idx] = val
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    # Recurse
                    if dfs(r_idx, c_idx + 1, min_last_for_next):
                        return True
                    
                    # Undo
                    for d in new_diffs:
                        used_diffs.discard(d)
                    rows[r_idx][c_idx] = 0 # Reset
                
                return False
        
        # Initialize rows
        # rows is already initialized to 0s
        
        # Start DFS
        # min_last_for_next for the first row is 0
        if dfs(0, 0, 0):
            return rows
        return None

    # Search loop
    # We start from current_best_M and go down.
    # We will keep the best solution found.
    
    # To avoid getting stuck in a very difficult M, we use a node limit.
    node_limit = 200000
    
    # We'll iterate M from current_best_M - 1 down to lower_bound.
    for M in range(current_best_M - 1, lower_bound - 1, -1):
        if time_left() < 5:
            break
        
        # Try to solve for M
        sol = solve_with_limit(M, node_limit)
        
        if sol is not None:
            best_rows = sol
            current_best_M = M
            # We can continue to lower M to see if we can do better.
            # But we should check if we have time.
    
    return best_rows
