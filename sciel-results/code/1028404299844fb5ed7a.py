# EVOLVE-BLOCK-START
import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    """
    # Lower bound estimation
    target = n_rows * k * (k + 1) // 2
    m_lb = 1
    while m_lb * (m_lb + 1) // 2 < target:
        m_lb += 1
    
    best_rows = None
    best_max = float('inf')
    
    start_time = time.time()
    time_limit = 110.0
    
    # Backtracking search for minimal M
    # We try M from m_lb upwards
    # We limit the number of nodes to prevent hanging
    
    node_limit = 200000 # Adjust based on performance
    
    for M in range(m_lb, 200):
        if time.time() - start_time > time_limit:
            break
            
        result = _backtrack_solve(n_rows, k, M, node_limit)
        
        if result:
            best_rows = result
            best_max = M
            break
            
    if best_rows is None:
        # Fallback to greedy if backtracking failed or timed out
        best_rows = _greedy_deterministic(n_rows, k)
        if best_rows:
            best_max = max(max(r) for r in best_rows)
            
    return best_rows

def _backtrack_solve(n_rows, k, M, node_limit):
    """
    Backtracking search for a valid configuration with max entry <= M.
    Returns list of rows or None.
    """
    used_diffs = set()
    rows = []
    
    state = {'nodes': 0, 'limit': node_limit}
    
    def solve(row_idx, col_idx, curr_row):
        if state['nodes'] > state['limit']:
            return False
        state['nodes'] += 1
        
        if row_idx == n_rows:
            return True
            
        if col_idx == 0:
            return solve(row_idx, 1, [0])
            
        last_val = curr_row[-1]
        
        for x in range(last_val + 1, M + 1):
            valid = True
            new_diffs = []
            for a in curr_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            curr_row.append(x)
            used_diffs.update(new_diffs)
            
            if col_idx == k:
                rows.append(curr_row[:])
                if solve(row_idx + 1, 0, []):
                    return True
                rows.pop()
            else:
                if solve(row_idx, col_idx + 1, curr_row):
                    return True
            
            curr_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False
        
    if solve(0, 0, []):
        return rows
    return None

def _greedy_deterministic(n_rows, k):
    """Original greedy construction."""
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

# EVOLVE-BLOCK-END
