import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer for safety

    def greedy_construct():
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

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    # If k is small, maybe greedy is already optimal?
    # For (7,5), greedy gives 112.
    
    diffs_per_row = k * (k + 1) // 2
    
    def solve_with_rows(M):
        used_diffs = set()
        rows = []
        
        def dfs(row_idx, current_row):
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return list(rows)
            
            if len(current_row) == k + 1:
                if rows:
                    if current_row[-1] < rows[-1][-1]:
                        return None
                
                rows.append(current_row)
                res = dfs(row_idx + 1, [0])
                if res:
                    return res
                rows.pop()
                return None
            
            last = current_row[-1]
            r = len(current_row)
            
            # Pruning: Check if enough differences remain
            # Diffs needed for current row completion
            # Elements left to add: k + 1 - r
            # Next element adds r diffs
            # ...
            # Last element adds k diffs
            need_curr = 0
            for j in range(r, k + 1):
                need_curr += j
                
            need_rem = (n_rows - row_idx - 1) * diffs_per_row
            total_need = need_curr + need_rem
            total_avail = M * (M - 1) // 2 - len(used_diffs)
            
            if total_avail < total_need:
                return None
            
            # Heuristic: Try x from last+1 to M
            # We can also try to pick x such that it doesn't "waste" small differences
            # But simple iteration is okay.
            
            for x in range(last + 1, M + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                used_diffs.update(new_diffs)
                current_row.append(x)
                
                res = dfs(row_idx, current_row)
                if res:
                    return res
                
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return None
        
        return dfs(0, [0])

    # Try to find a better solution
    # Start from lower bound
    # Lower bound M such that M*(M-1)/2 >= n_rows * diffs_per_row
    lb = 1
    while lb * (lb - 1) // 2 < n_rows * diffs_per_row:
        lb += 1
        
    # Try from lb to best_scope - 1
    # If lb is very small, we might waste time.
    # But lb is 15 for (7,5).
    # We know 112 works.
    # We try 105 to 111? Or lb to 111?
    # The prompt says counting bound is 105.
    # So we start from 105.
    start_M = max(lb, 105)
    
    for M in range(start_M, best_scope):
        if time.time() > deadline - 5:
            break
        
        try:
            result = solve_with_rows(M)
            if result:
                return result
        except Exception:
            break

    return best_rows
