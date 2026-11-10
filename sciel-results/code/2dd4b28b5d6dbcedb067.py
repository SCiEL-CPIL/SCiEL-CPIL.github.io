import time
import random
import sys

sys.setrecursionlimit(10000)

DEADLINE = 0
TIME_LIMIT = 115

def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT
    
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0
    
    generic_lb = n_rows * (k * (k + 1) // 2)
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = generic_lb

    start_m = max(lower_bound, generic_lb)
    
    for M in range(start_m, baseline_scope + 1):
        if time.time() > DEADLINE:
            break
        
        solution = solve_optimized(n_rows, k, M)
        if solution is not None:
            return solution
            
    return baseline_rows

def greedy_construct(n_rows, k):
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

def solve_optimized(n_rows, k, M):
    diffs_per_row = k * (k + 1) // 2
    used_mask = 0
    rows = []
    
    def build_row(current_row, current_row_mask, rows_remaining, randomize_order):
        if time.time() > DEADLINE:
            return False
            
        if len(current_row) == k + 1:
            rows.append(list(current_row))
            nonlocal used_mask
            used_mask |= current_row_mask
            return True
            
        remaining_needed = (rows_remaining + 1) * diffs_per_row - current_row_mask.bit_count()
        available = M - used_mask.bit_count()
        
        if available < remaining_needed:
            return False
            
        last_val = current_row[-1]
        candidates = []
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                if d > M:
                    valid = False
                    break
                bit = 1 << d
                if (used_mask & bit) or (current_row_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            if valid:
                candidates.append((x, new_mask))
                
        if not candidates:
            return False
            
        if randomize_order:
            random.shuffle(candidates)
            
        for x, new_mask in candidates:
            current_row.append(x)
            if build_row(current_row, current_row_mask | new_mask, rows_remaining - 1, randomize_order):
                return True
            current_row.pop()
            
        return False

    def run_search(randomize):
        nonlocal used_mask, rows
        used_mask = 0
        rows = []
        
        for i in range(n_rows):
            if time.time() > DEADLINE:
                return False
            if not build_row([0], 0, n_rows - i - 1, randomize):
                return False
        return True

    if run_search(False):
        return rows
        
    for _ in range(5):
        if time.time() > DEADLINE:
            return None
        if run_search(True):
            return rows
            
    return None
