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
    
    # 1. Fast baseline
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0
    
    # 2. Search from lower bound up to baseline_scope
    generic_lb = n_rows * (k * (k + 1) // 2)
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = generic_lb
    
    start_m = max(lower_bound, generic_lb)
    
    for M in range(start_m, baseline_scope + 1):
        if time.time() > DEADLINE:
            break
        solution = solve_fast(n_rows, k, M)
        if solution is not None:
            return solution
            
    return baseline_rows

def greedy_construct(n_rows, k):
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

def solve_fast(n_rows, k, M):
    """
    Fast backtracking solver using integer bitmasks for difference tracking.
    Uses aggressive pruning and randomized restarts.
    """
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Quick feasibility check
    if M < total_diffs_needed:
        return None
    
    used_mask = 0
    rows = []
    
    def pc(x):
        # Fast popcount using Python's int.bit_count() (Python 3.10+)
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
    
    def build_row(row, row_mask, row_idx, randomize):
        nonlocal used_mask, rows
        
        if time.time() > DEADLINE:
            return False
            
        if len(row) == k + 1:
            rows.append(list(row))
            used_mask |= row_mask
            return True
            
        # Pruning: remaining differences must fit
        # Diffs already in this row: pc(row_mask)
        # Diffs used globally: pc(used_mask)
        # Remaining rows to build: n_rows - row_idx - 1
        # Total remaining diffs needed: (n_rows - row_idx - 1) * diffs_per_row + (diffs_per_row - pc(row_mask))
        remaining_rows = n_rows - row_idx - 1
        remaining_needed = remaining_rows * diffs_per_row + (diffs_per_row - pc(row_mask))
        available = M - pc(used_mask)
        
        if available < remaining_needed:
            return False
            
        last = row[-1]
        candidates = []
        
        # Generate valid candidates
        for x in range(last + 1, M + 1):
            valid = True
            new_mask = 0
            for a in row:
                d = x - a
                if d > M:
                    valid = False
                    break
                bit = 1 << d
                if (used_mask & bit) or (row_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            if valid:
                candidates.append((x, new_mask))
                
        if not candidates:
            return False
            
        # Order candidates
        if randomize:
            random.shuffle(candidates)
        else:
            # Try smallest first for tight packing
            pass
            
        for x, new_mask in candidates:
            row.append(x)
            if build_row(row, row_mask | new_mask, row_idx, randomize):
                return True
            row.pop()
            
        return False
    
    def run_attempt(randomize, seed=None):
        nonlocal used_mask, rows
        if seed is not None:
            random.seed(seed)
        used_mask = 0
        rows = []
        
        for i in range(n_rows):
            if time.time() > DEADLINE:
                return False
            if not build_row([0], 0, i, randomize):
                return False
        return True
    
    # Try deterministic first
    if run_attempt(False):
        return rows
    
    # Try several randomized restarts
    for attempt in range(20):
        if time.time() > DEADLINE:
            break
        seed = random.randint(0, 10**9)
        if run_attempt(True, seed):
            return rows
    
    return None
