import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Fast Greedy Baseline
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return []
    
    baseline_max = max(max(row) for row in baseline_rows)
    
    # 2. Determine Lower Bound
    diffs_per_row = k * (k + 1) // 2
    generic_lb = n_rows * diffs_per_row
    
    # Specific known lower bound for D(7,5) is 105
    if n_rows == 7 and k == 5:
        lower_bound = max(generic_lb, 105)
    else:
        lower_bound = generic_lb
        
    # Ensure lower bound is at least the max element required by the greedy start (0)
    lower_bound = max(lower_bound, k)

    # 3. Backtracking Search for Minimum M
    for M in range(lower_bound, baseline_max + 1):
        if time.time() > deadline:
            break
            
        found = False
        max_restarts = 10
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            solution = solve_backtrack(n_rows, k, M, randomize=(restart > 0), seed=restart, deadline=deadline)
            if solution is not None:
                return solution
                
    return baseline_rows

def greedy_construct(n_rows, k):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            new_mask = 0
            for a in row:
                d = x - a
                if d > 200:
                    valid = False
                    break
                bit = 1 << d
                if (used & bit) or (new_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            
            if valid:
                row.append(x)
                used |= new_mask
        
        rows.append(row)
    return rows

def solve_backtrack(n_rows, k, M, randomize=False, seed=0, deadline=0):
    if randomize:
        random.seed(seed)
        
    diffs_per_row = k * (k + 1) // 2
    rows = []
    
    def dfs(row_idx, used_mask):
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            return True
            
        remaining_rows = n_rows - row_idx
        needed = remaining_rows * diffs_per_row
        available = M - bin(used_mask).count('1')
        if available < needed:
            return False
            
        # Try to build the next row
        row, new_used_mask = build_row(used_mask)
        if row is None:
            return False
            
        rows.append(row)
        if dfs(row_idx + 1, new_used_mask):
            return True
            
        rows.pop()
        return False

    def build_row(used_mask):
        current_row = [0]
        current_mask = 0
        result = extend_row(used_mask, current_row, current_mask)
        if result is not None:
            return current_row, used_mask | current_mask
        return None, None

    def extend_row(used_mask, current_row, current_mask):
        if time.time() > deadline:
            return False
            
        if len(current_row) == k + 1:
            return True
            
        last_val = current_row[-1]
        candidates = []
        
        # Heuristic: check if we can fit the remaining elements
        # Min next is last_val + 1.
        # Max possible value for last element is last_val + (k + 1 - len(current_row))
        if last_val + (k + 1 - len(current_row)) > M:
            return False

        for x in range(last_val + 1, M + 1):
            valid = True
            new_mask = 0
            for a in current_row:
                d = x - a
                if d > M:
                    valid = False
                    break
                bit = 1 << d
                if (used_mask & bit) or (current_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            
            if valid:
                candidates.append((x, new_mask))
                
        if not candidates:
            return False
            
        if randomize:
            random.shuffle(candidates)
            
        for x, new_mask in candidates:
            current_row.append(x)
            if extend_row(used_mask, current_row, current_mask | new_mask):
                return True
            current_row.pop()
            
        return False

    if dfs(0, 0):
        return [list(r) for r in rows]
    return None
