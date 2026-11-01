def construct(n_rows, k):
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * total_diffs_per_row
    lower_bound = total_diffs
    upper_bound = lower_bound + 100
    
    def check(M):
        used_diffs = set()
        rows = []
        node_count = 0
        max_nodes = 100000 # Limit to prevent timeout on hard instances
        
        def dfs(row_idx, current_row, all_used, current_row_diffs):
            nonlocal node_count
            node_count += 1
            if node_count > max_nodes:
                return False
            
            if current_row and current_row[-1] > M:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                new_all_used = all_used | current_row_diffs
                rows.append(current_row)
                
                if dfs(row_idx + 1, [0], new_all_used, set()):
                    return True
                
                rows.pop()
                return False
            
            last_val = current_row[-1] if current_row else 0
            
            # Optimization: If we have many rows left, we might want to prune more aggressively?
            # For now, simple DFS.
            
            for x in range(last_val + 1, M + 1):
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in all_used or d in current_row_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                new_row = current_row + [x]
                new_current_row_diffs = current_row_diffs | new_diffs
                
                if dfs(row_idx, new_row, all_used, new_current_row_diffs):
                    return True
            
            return False
        
        if dfs(0, [0], set(), set()):
            return rows
        return None

    for M in range(lower_bound, upper_bound + 1):
        res = check(M)
        if res is not None:
            return res
    
    # Fallback
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
