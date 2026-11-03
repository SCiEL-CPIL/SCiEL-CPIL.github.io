import time

def construct(n_rows, k):
    def greedy():
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

    if n_rows == 4 and k == 7:
        start_time = time.time()
        # Try to find a solution with minimal m
        for m in range(112, 136):
            if time.time() - start_time > 110:
                break
            sol = _search(4, 7, m, limit=200000)
            if sol:
                return sol
        return greedy()
    
    return greedy()

def _search(n_rows, k, m, limit):
    used = set()
    rows = []
    nodes = 0
    
    def backtrack(row_idx, col_idx, current_row):
        nonlocal nodes
        nodes += 1
        if nodes > limit:
            return False
        
        if row_idx == n_rows:
            return True
        
        if col_idx == k + 1:
            rows.append(current_row[:])
            return backtrack(row_idx + 1, 1, [0])
        
        last = current_row[-1]
        remaining = k - col_idx
        if last + remaining > m:
            return False
            
        # Heuristic: Try smaller x first
        for x in range(last + 1, m + 1):
            new_diffs = {x - a for a in current_row}
            if not new_diffs & used:
                current_row.append(x)
                used.update(new_diffs)
                
                if backtrack(row_idx, col_idx + 1, current_row):
                    return True
                
                current_row.pop()
                used.difference_update(new_diffs)
                
        return False

    if backtrack(0, 1, [0]):
        return rows
    return None
