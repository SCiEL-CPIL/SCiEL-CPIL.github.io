import time

_DEADLINE_MARGIN = 10.0
_start_time = None

def _time_exceeded():
    if _start_time is None:
        return False
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)

def _solve_scope(n_rows, k, scope):
    diff_mask = 0
    rows = [[0] * (k + 1) for _ in range(n_rows)]
    
    def backtrack(r, pos):
        if _time_exceeded():
            return False
            
        if r == n_rows:
            return True
            
        if pos == k + 1:
            return backtrack(r + 1, 1)
            
        lb = rows[r][pos - 1] + 1
        if pos == 1 and r > 0:
            lb = max(lb, rows[r-1][1] + 1)
            
        for v in range(lb, scope + 1):
            valid = True
            new_diffs_mask = 0
            for i in range(pos):
                d = v - rows[r][i]
                if (diff_mask >> d) & 1:
                    valid = False
                    break
                new_diffs_mask |= (1 << d)
                
            if not valid:
                continue
                
            old_mask = diff_mask
            diff_mask = old_mask | new_diffs_mask
            rows[r][pos] = v
            
            if backtrack(r, pos + 1):
                return True
                
            diff_mask = old_mask
            
        return False
        
    if backtrack(0, 1):
        return rows
    return None

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        last = 0
        while len(row) <= k:
            x = last + 1
            while True:
                new_diffs = [x - a for a in row]
                if not any(d in used for d in new_diffs):
                    break
                x += 1
            row.append(x)
            used.update(new_diffs)
            last = x
        rows.append(row)
    return rows

def construct(n_rows, k):
    global _start_time
    _start_time = time.time()
    
    greedy_rows = _greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in greedy_rows)
    best_rows = greedy_rows
    
    count = n_rows * k * (k + 1) // 2
    lb = count
    
    low = lb
    high = best_scope
    
    while low < high and not _time_exceeded():
        mid = (low + high) // 2
        res = _solve_scope(n_rows, k, mid)
        if res:
            high = mid
            best_scope = mid
            best_rows = res
        else:
            low = mid + 1
            
    return best_rows
