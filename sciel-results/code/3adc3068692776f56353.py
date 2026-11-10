import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Lower bound on M
    num_diffs_per_row = (k + 1) * k // 2
    lb = n_rows * num_diffs_per_row
    
    # We will try M from lb upwards
    # But we also need to handle the case where no solution is found quickly
    # We can interleave increasing M with random restarts
    
    best_solution = None
    best_m = float('inf')
    
    # Helper to validate and return rows
    def validate(rows):
        used = set()
        for row in rows:
            diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used or d in diffs:
                        return False
                    diffs.add(d)
            used.update(diffs)
        return True

    # Backtracking search for a given M
    def search(M):
        used_diffs = set()
        rows = []
        
        def dfs(row_idx, current_row, current_diffs):
            nonlocal used_diffs, rows, best_solution, best_m
            
            if time.time() > deadline:
                return False
                
            if row_idx == n_rows:
                # Found a solution
                best_solution = [r[:] for r in rows]
                best_m = M
                return True
                
            if len(current_row) == k + 1:
                # Check if current row is valid
                # current_diffs contains all diffs in current_row
                if not (current_diffs & used_diffs):
                    # Save state
                    used_diffs.update(current_diffs)
                    rows.append(current_row[:])
                    
                    if dfs(row_idx + 1, [], set()):
                        return True
                        
                    # Backtrack
                    rows.pop()
                    used_diffs.difference_update(current_diffs)
                return False
            
            # Determine next value x
            last = current_row[-1] if current_row else -1
            # x must be > last and <= M
            # Also, we should ensure that we can still fit the remaining elements
            # Min possible next value is last + 1
            # Max possible next value is M
            
            # Heuristic: Try values in a randomized order or specific order
            # For finding a valid solution, randomization helps avoid dead ends
            # But we want to find a solution with small M, so we are fixing M here
            
            # To speed up, we can skip x if it creates a conflict immediately
            # We iterate x from last+1 to M
            
            # Optimization: If we are in the first row, we can try to be "tight"
            # But randomization is generally better for CSP
            
            candidates = list(range(last + 1, M + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = set()
                conflict = False
                for a in current_row:
                    d = x - a
                    if d in used_diffs or d in current_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Add x
                current_row.append(x)
                current_diffs.update(new_diffs)
                
                if dfs(row_idx, current_row, current_diffs):
                    return True
                
                # Backtrack
                current_row.pop()
                current_diffs.difference_update(new_diffs)
                
        return dfs(0, [0], set())

    # Try increasing M
    # We can limit the number of M values we try to avoid spending too much time on impossible M
    # Or just try M=lb, lb+1, ... until found or time out
    
    # To improve odds, we can do multiple restarts for each M
    # But for small M, the search space is constrained.
    
    # Let's try M from lb to lb + 20
    # If not found, we might need to go higher, but we run out of time.
    # The prompt says best known is 112 for (7,5). LB is 105.
    # So we should check 105..112.
    
    for m in range(lb, lb + 30):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this M
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
            random.seed(restart) # Or use a global random state
            if search(m):
                found = True
                break
        
        if found:
            return best_solution
            
    # If no solution found with M search, fallback to greedy
    # Greedy ensures we return *something* valid
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
        
    return rows
