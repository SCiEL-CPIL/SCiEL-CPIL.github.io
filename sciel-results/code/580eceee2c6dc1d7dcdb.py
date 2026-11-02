import time
import copy

def construct(n_rows, k):
    """
    Constructs a difference triangle set (n_rows, k).
    Returns a list of n_rows rows, each of length k+1, starting at 0.
    Minimizes the largest entry.
    """
    if n_rows == 0:
        return []
    
    # Greedy initialization to get a good starting bound
    def greedy():
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
        return rows

    init_rows = greedy()
    best_scope = max(r[-1] for r in init_rows)
    best_rows = copy.deepcopy(init_rows)
    
    # Backtracking search
    MAX_VAL = 200  # Safe upper bound for small k
    used = bytearray(MAX_VAL + 1)
    rows = []
    current_row = []
    start_time = time.time()
    timeout = 110  # Leave 10s margin for Python overhead
    
    def dfs(r, c):
        nonlocal best_scope, best_rows, used, rows, current_row
        
        # Time check
        if time.time() - start_time > timeout:
            return True  # Signal timeout
        
        if r == n_rows:
            scope = max(row[-1] for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = copy.deepcopy(rows)
            return False
        
        if c == 0:
            current_row = [0]
            return dfs(r, 1)
        
        prev = current_row[-1]
        # Prune: x must be <= best_scope
        max_x = best_scope
        
        # Lexicographic ordering: rows[r-1] <=_lex current_row
        # If r > 0, check against rows[r-1]
        prev_row = rows[r-1] if r > 0 else None
        
        for x in range(prev + 1, max_x + 1):
            # Lexicographic check
            if prev_row is not None:
                if c < len(prev_row):
                    if x < prev_row[c]:
                        continue
                    # If x == prev_row[c], it's okay (might be equal so far)
                    # If x > prev_row[c], it's okay (already greater)
            
            # Check conflicts
            conflict = False
            new_diffs = []
            for a in current_row:
                d = x - a
                if d > MAX_VAL:
                    conflict = True
                    break
                if used[d]:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if conflict:
                continue
            
            # Accept x
            current_row.append(x)
            for d in new_diffs:
                used[d] = 1
            
            if c == k:
                rows.append(current_row.copy())
                if dfs(r + 1, 0):
                    # Timeout
                    rows.pop()
                    current_row.pop()
                    for d in new_diffs:
                        used[d] = 0
                    return True
                rows.pop()
            else:
                if dfs(r, c + 1):
                    current_row.pop()
                    for d in new_diffs:
                        used[d] = 0
                    return True
            
            # Backtrack
            current_row.pop()
            for d in new_diffs:
                used[d] = 0
        
        return False

    dfs(0, 0)
    return best_rows
