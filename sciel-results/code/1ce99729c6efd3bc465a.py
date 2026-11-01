import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    min_m = n_rows * k * (k+1) // 2
    
    for M in range(min_m, min_m + 20):
        if time.time() > deadline:
            break
            
        used_diffs = 0  # Bitmask of used differences
        rows = []
        
        def backtrack(row_idx, current_row, last_val, current_diffs):
            if time.time() > deadline:
                return False, None, 0
            
            if row_idx == n_rows:
                return True, rows, used_diffs
            
            remaining = (k + 1) - len(current_row)
            
            if remaining == 0:
                # Row is complete
                rows.append(list(current_row))
                new_used = used_diffs | current_diffs
                # Recurse to next row
                ok, res_rows, res_used = backtrack(row_idx + 1, [0], 0, 0)
                if ok:
                    return True, res_rows, res_used
                else:
                    rows.pop()
                    return False, None, 0
            
            # Prune: not enough numbers left
            if M - last_val < remaining:
                return False, None, 0
            
            # Iterate x
            # We can try to optimize the order of x.
            # For now, increasing.
            for x in range(last_val + 1, M + 1):
                valid = True
                new_diffs_mask = current_diffs
                for a in current_row:
                    d = x - a
                    # d is always >= 1
                    bit = 1 << d
                    if (new_diffs_mask | used_diffs) & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                
                if valid:
                    new_row = current_row + [x]
                    ok, res_rows, res_used = backtrack(row_idx, new_row, x, new_diffs_mask)
                    if ok:
                        return True, res_rows, res_used
            
            return False, None, 0
        
        ok, res_rows, _ = backtrack(0, [0], 0, 0)
        
        if ok:
            return res_rows
        
    # Fallback: Greedy
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
# EVOLVE-BLOCK-END
