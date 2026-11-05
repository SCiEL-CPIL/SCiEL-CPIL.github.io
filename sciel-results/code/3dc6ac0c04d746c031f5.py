"""Difference triangle set (7,5) — Backtracking Solver."""
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences
    within each row are distinct across all rows.
    
    Uses a deterministic backtracking search with forward checking.
    """
    if n_rows <= 0 or k <= 0:
        return []
    
    # Total number of differences needed: n_rows * (k+1)*k/2
    num_diffs = n_rows * (k + 1) * k // 2
    
    # Lower bound on largest entry M:
    # The available differences are 1..M. We need num_diffs distinct ones.
    # So M*(M+1)/2 >= num_diffs.
    low = 1
    while low * (low + 1) // 2 < num_diffs:
        low += 1
    
    # We search for the minimum M starting from the lower bound.
    # The known best is 112 for (7,5). The lower bound is 105.
    # We try M = low, low+1, ... until we find a solution or time out.
    
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    best_rows = None
    best_scope = float('inf')
    
    for M in range(low, 200):  # Cap search range reasonably
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a solution with max entry M
        rows = _solve_backtracking(n_rows, k, M, start_time, time_limit)
        
        if rows is not None:
            best_rows = rows
            best_scope = M
            break  # Since we iterate M upwards, first solution is minimal
    
    if best_rows is None:
        # Fallback to a simple greedy if backtracking failed (should not happen for this size)
        best_rows = _construct_simple_greedy(n_rows, k)
        
    return best_rows

def _solve_backtracking(n_rows, k, max_val, start_time, time_limit):
    """
    Backtracking solver to find n_rows of size k+1 with max entry <= max_val
    and all differences distinct.
    """
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, current_row):
        if time.time() - start_time > time_limit:
            return False
            
        if row_idx == n_rows:
            return True
            
        if len(current_row) == k + 1:
            rows.append(current_row)
            success = backtrack(row_idx + 1, [0])
            if success:
                return True
            rows.pop()
            return False
            
        last_val = current_row[-1]
        remaining = (k + 1) - len(current_row)
        max_next = max_val - (remaining - 1)
        
        # Heuristic: Try smaller values first to keep the row compact
        for x in range(last_val + 1, max_next + 1):
            valid = True
            new_diffs = []
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            current_row.append(x)
            used_diffs.update(new_diffs)
            
            if backtrack(row_idx, current_row):
                return True
                
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False
        
    # Start
    if backtrack(0, [0]):
        return rows
    return None

def _construct_simple_greedy(n_rows, k):
    """Fallback greedy approach."""
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
