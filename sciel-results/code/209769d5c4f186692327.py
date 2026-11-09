import time

def construct(n_rows, k):
    total_diffs = n_rows * k * (k+1) // 2
    M_min = total_diffs
    
    best_solution = None
    best_M = float('inf')
    
    diffs_per_row = k * (k+1) // 2
    
    def solve(M, time_limit):
        start = time.time()
        used = set()
        rows = []
        
        def bt(row_idx, current_row, used, rows):
            if time.time() - start > time_limit:
                return False
            
            if row_idx == n_rows:
                nonlocal best_solution, best_M
                if M < best_M:
                    best_M = M
                    best_solution = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                new_rows = rows + [current_row]
                return bt(row_idx + 1, [0], used, new_rows)
            
            L = len(current_row)
            remaining_diffs_current = sum(range(L, k+1))
            future_diffs = (n_rows - row_idx - 1) * diffs_per_row
            total_needed = remaining_diffs_current + future_diffs
            if M - len(used) < total_needed:
                return False
            
            last = current_row[-1]
            # Heuristic: Try smaller x first
            for x in range(last + 1, M + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used:
                    continue
                
                used.update(new_diffs)
                new_row = current_row + [x]
                
                if bt(row_idx, new_row, used, rows):
                    return True
                
                used.difference_update(new_diffs)
            
            return False
        
        return bt(0, [0], used, rows)

    # Try M from M_min to M_min + 20
    # We have 120s.
    # Let's try 105 with 20s, 106 with 20s, etc.
    # If we find a solution, we return it.
    
    max_M_to_try = M_min + 20
    for M in range(M_min, max_M_to_try + 1):
        # Check if we have time left?
        # We can't easily check global time here without passing it.
        # But solve() has its own limit.
        
        if solve(M, 15):
            if best_solution:
                return best_solution
    
    if best_solution:
        return best_solution
        
    # Fallback to greedy
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
