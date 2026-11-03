import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    L = k + 1
    N = n_rows
    D_row = L * (L - 1) // 2
    D_total = N * D_row
    
    # 1. Greedy Solution to get an upper bound
    def get_greedy():
        used = set()
        rows = []
        for _ in range(N):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    greedy_rows = get_greedy()
    best_scope = max(max(r) for r in greedy_rows)
    
    # If k is very small, greedy is likely optimal or close
    if D_total == 0:
        return greedy_rows

    # 2. Backtracking Search with Bitmask
    M_min = D_total
    M_max = best_scope
    
    # We search for the smallest M in [M_min, M_max]
    for M in range(M_min, M_max + 1):
        if time.time() > deadline:
            break
            
        # State for backtracking
        # We use a list of rows, current row, used mask, and row index
        rows = []
        current_row = [0]
        used_mask = 0
        row_idx = 0
        max_val = 0
        
        # Pre-compute bit masks for differences? 
        # d goes up to M. M is small.
        # bit = 1 << d
        
        def dfs():
            nonlocal rows, current_row, used_mask, row_idx, max_val
            
            if time.time() > deadline:
                return False
                
            # Base case: All rows constructed
            if row_idx == N:
                return True
            
            # If current row is full
            if len(current_row) == L:
                rows.append(current_row[:])
                row_idx += 1
                current_row = [0]
                max_val = 0
                
                if dfs():
                    return True
                
                # Backtrack
                rows.pop()
                row_idx -= 1
                current_row = [0]
                max_val = 0
                return False
            
            # Determine next value
            rem = L - len(current_row)
            start_val = max_val + 1
            
            # Symmetry breaking: 
            # If this is not the first row, the first non-zero element (index 1)
            # must be >= the first non-zero element of the previous row.
            if row_idx > 0 and len(current_row) == 1:
                prev_min = rows[-1][1]
                if start_val < prev_min:
                    start_val = prev_min
            
            # Pruning: Can we fit the remaining elements?
            # We need 'rem - 1' more elements.
            # The smallest they can be is start_val, start_val+1, ...
            # So the largest will be at least start_val + rem - 1.
            if start_val + rem - 1 > M:
                return False
            
            for x in range(start_val, M + 1):
                new_diffs = [x - a for a in current_row]
                mask = 0
                valid = True
                for d in new_diffs:
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    mask |= bit
                
                if not valid:
                    continue
                
                # Make move
                current_row.append(x)
                max_val = x
                used_mask |= mask
                
                if dfs():
                    return True
                    
                # Undo move
                current_row.pop()
                max_val = current_row[-1]
                used_mask &= ~mask
                
            return False
        
        if dfs():
            return rows

    # If no better solution found or timeout, return greedy
    return greedy_rows
