import time

def construct(n_rows, k):
    start_time = time.time()
    timeout = 145
    
    lower_bound = n_rows * k * (k + 1) // 2
    
    def search(M):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_max):
            if time.time() - start_time > timeout:
                return None
            
            if row_idx == n_rows:
                return rows[:]
            
            # If first row, fix first element to 1
            if row_idx == 0:
                # Try starting with [0, 1]
                if 1 > M:
                    return None
                if 1 in used_diffs:
                    return None
                used_diffs.add(1)
                res = backtrack_row(1, [0, 1], {1}, 1)
                if res:
                    return res
                used_diffs.remove(1)
                return None
            
            return backtrack_row(row_idx, [0], set(), current_max)
        
        def backtrack_row(row_idx, prefix, current_diffs, current_max):
            if time.time() - start_time > timeout:
                return None
            
            if len(prefix) == k + 1:
                # Row complete
                if prefix[-1] < current_max:
                    return None
                
                # Check if all diffs in current_diffs are unused globally
                # (They are checked during construction, but just in case)
                # Actually, we check against used_diffs at each step.
                
                rows.append(prefix)
                used_diffs.update(current_diffs)
                
                res = backtrack(row_idx + 1, prefix[-1])
                
                used_diffs.difference_update(current_diffs)
                
                if res:
                    return res
                return None
            
            last = prefix[-1]
            remaining = (k + 1) - len(prefix)
            
            if last + remaining > M:
                return None
            
            # Heuristic: Try smaller x first
            for x in range(last + 1, M + 1):
                new_diffs = {x - a for a in prefix}
                
                if new_diffs & used_diffs:
                    continue
                if new_diffs & current_diffs:
                    continue
                
                res = backtrack_row(row_idx, prefix + [x], current_diffs | new_diffs, current_max)
                if res:
                    return res
            
            return None
        
        return backtrack(0, 0)

    best_rows = None
    for M in range(lower_bound, 200):
        if time.time() - start_time > timeout:
            break
        res = search(M)
        if res:
            best_rows = res
            break
    
    if best_rows is None:
        # Fallback
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
        best_rows = rows
        
    return best_rows
