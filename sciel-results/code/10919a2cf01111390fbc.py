import sys

def construct(n_rows, k):
    sys.setrecursionlimit(10000)
    
    lower_bound = n_rows * k * (k + 1) // 2
    
    # We try a small range around the lower bound
    # If the best known is 112 and lower bound is 105, we try 105..115
    for M in range(lower_bound, lower_bound + 20):
        result = _solve_with_M(n_rows, k, M)
        if result is not None:
            return result
            
    # Fallback
    return _greedy_construct(n_rows, k)

def _solve_with_M(n_rows, k, M):
    used_mask = 0
    rows = []
    
    def backtrack_row(row_idx, current_row, used_mask, min_max_for_this_row):
        if len(current_row) == k + 1:
            row_max = current_row[-1]
            if row_max < min_max_for_this_row:
                return None
            
            rows.append(list(current_row))
            
            if row_idx == n_rows - 1:
                return rows[:]
            
            res = backtrack_row(row_idx + 1, [0], used_mask, row_max)
            if res is not None:
                return res
            else:
                rows.pop()
                return None
        
        last_x = current_row[-1]
        next_start = last_x + 1
        
        if len(current_row) == k:
            next_start = max(next_start, min_max_for_this_row)
            
        s = len(current_row)
        r = k + 1 - s
        new_diffs_this_row = r * s + r * (r - 1) // 2
        future_rows = n_rows - row_idx - 1
        diffs_per_future_row = k * (k + 1) // 2
        new_diffs_future = future_rows * diffs_per_future_row
        total_needed = new_diffs_this_row + new_diffs_future
        
        used_count = bin(used_mask).count('1')
        available = M - used_count
        
        if available < total_needed:
            return None
        
        for x in range(next_start, M + 1):
            new_diffs = [x - a for a in current_row]
            
            conflict = False
            new_mask = 0
            for d in new_diffs:
                bit = 1 << d
                if used_mask & bit:
                    conflict = True
                    break
                new_mask |= bit
            
            if conflict:
                continue
            
            res = backtrack_row(row_idx, current_row + [x], used_mask | new_mask, min_max_for_this_row)
            if res is not None:
                return res
        
        return None

    return backtrack_row(0, [0], 0, 0)

def _greedy_construct(n_rows, k):
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
