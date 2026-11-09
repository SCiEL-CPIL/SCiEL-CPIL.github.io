import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Helper to check if a row is valid internally (distinct diffs)
    # and return the set of differences
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return None
                diffs.add(d)
        return diffs

    # Greedy construction to get a baseline
    def greedy_construct(n_rows, k):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows

    # Initial solution
    best_rows = greedy_construct(n_rows, k)
    best_max = max(max(r) for r in best_rows)
    
    # If we run out of time, return the greedy solution
    if time.time() > deadline - 5:
        return best_rows

    # Backtracking search for a solution with max value <= M
    def search_solution_local(M_limit):
        used_diffs = set()
        current_rows = []
        
        def backtrack(row_idx):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            row = [0]
            
            def build_row(pos, last_val):
                if time.time() > deadline:
                    return None
                
                if pos == k + 1:
                    return row
                
                start = last_val + 1
                # Heuristic: We can't just go to M_limit. 
                # We need to leave room for the remaining elements.
                # Max possible value for current element x is M_limit - (k - pos)
                # because we need k-pos more elements, each at least +1.
                max_x = M_limit - (k - pos)
                
                for x in range(start, max_x + 1):
                    new_diffs = {x - a for a in row}
                    
                    # Check internal conflicts
                    # Compute current row diffs
                    current_row_diffs = set()
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            current_row_diffs.add(row[j] - row[i])
                            
                    if new_diffs & current_row_diffs:
                        continue
                    if new_diffs & used_diffs:
                        continue
                    
                    row.append(x)
                    res = build_row(pos + 1, x)
                    if res is not None:
                        return res
                    row.pop()
                
                return None
            
            res = build_row(1, 0)
            if res is not None:
                row_diffs = get_diffs(res)
                used_diffs.update(row_diffs)
                current_rows.append(res)
                
                if backtrack(row_idx + 1):
                    return True
                
                current_rows.pop()
                used_diffs.difference_update(row_diffs)
            
            return False
        
        if backtrack(0):
            return current_rows
        return None

    # Try to find a solution with smaller M
    # We iterate M downwards from best_max - 1
    lower_bound = 112
    
    for M in range(best_max - 1, lower_bound - 1, -1):
        if time.time() > deadline - 10:
            break
        
        res = search_solution_local(M)
        if res is not None:
            best_rows = res
            best_max = M

    return best_rows
