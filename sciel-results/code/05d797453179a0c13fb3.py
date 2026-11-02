import time

def construct(n_rows, k):
    # Calculate lower bound M
    total_diffs = n_rows * k * (k + 1) // 2
    M_lb = 1
    while M_lb * (M_lb + 1) // 2 < total_diffs:
        M_lb += 1
    
    start_time = time.time()
    
    # Try to find optimal solution via backtracking
    # We search M from M_lb upwards
    # To prevent hanging, we limit the number of M values to try or total time
    max_M_to_try = M_lb + 20 # Heuristic limit
    
    for M in range(M_lb, max_M_to_try):
        if time.time() - start_time > 100:
            break
        solution = solve(n_rows, k, M)
        if solution:
            return solution
            
    # Fallback to greedy if no optimal found or timeout
    return greedy_construct(n_rows, k)

def solve(n_rows, k, M):
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx):
        if row_idx == n_rows:
            return True
        
        # Symmetry breaking: rows must be lexicographically non-decreasing
        prev_row = rows[row_idx - 1] if row_idx > 0 else None
        
        # We need to construct a row of length k+1 starting with 0
        # Elements must be in 0..M
        # Differences must not conflict with used_diffs
        
        current_row = [0]
        local_diffs = set()
        
        def build_row(idx, last_val):
            if idx == k + 1:
                # Check symmetry
                if prev_row is not None and current_row < prev_row:
                    return False
                
                rows.append(current_row[:])
                for d in local_diffs:
                    used_diffs.add(d)
                
                if backtrack(row_idx + 1):
                    return True
                
                rows.pop()
                for d in local_diffs:
                    used_diffs.discard(d)
                return False
            
            # Try next value x
            # Heuristic: Try values that allow future placement?
            # For now, iterate from last_val + 1 to M
            for x in range(last_val + 1, M + 1):
                # Check conflicts
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Pruning: If we are at idx, we have k+1-idx elements left to place.
                # The smallest possible next value is x+1.
                # We need to fit k+1-idx-1 more elements after x.
                # This requires at least (k+1-idx-1) * 1 space? No, they can be close.
                # But they must be distinct.
                # A simple prune: if x + (k + 1 - idx) > M: continue?
                # No, the elements don't have to be consecutive.
                # But if x is too large, we might not have enough room for the remaining elements?
                # Actually, we just need k+1-idx-1 distinct integers > x up to M.
                # So we need M - x >= k + 1 - idx - 1.
                if M - x < k + 1 - idx - 1:
                    continue
                
                current_row.append(x)
                local_diffs.update(new_diffs)
                
                if build_row(idx + 1, x):
                    return True
                
                current_row.pop()
                local_diffs.difference_update(new_diffs)
                
            return False
        
        return build_row(1, 0)
    
    if backtrack(0):
        return rows[:]
    return None

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
