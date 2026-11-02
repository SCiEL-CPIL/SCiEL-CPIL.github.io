import time

_START_TIME = time.time()
_TIME_LIMIT = 110.0

def construct(n_rows, k):
    global _START_TIME
    
    lower_bound = n_rows * k * (k + 1) // 2
    
    # Fallback greedy
    def _greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    greedy_rows = _greedy()
    greedy_max = max(max(r) for r in greedy_rows)
    
    if greedy_max == lower_bound:
        return greedy_rows

    deadline = _START_TIME + _TIME_LIMIT
    
    def solve_with_max(M):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx):
            if time.time() > deadline - 2:
                return False
            if row_idx == n_rows:
                return True
            
            def fill_row(current_row, pos):
                if time.time() > deadline - 2:
                    return False
                if pos == k + 1:
                    # Add diffs
                    for i in range(len(current_row)):
                        for j in range(i + 1, len(current_row)):
                            used_diffs.add(current_row[j] - current_row[i])
                    rows.append(current_row[:])
                    
                    if backtrack(row_idx + 1):
                        return True
                    
                    rows.pop()
                    for i in range(len(current_row)):
                        for j in range(i + 1, len(current_row)):
                            used_diffs.remove(current_row[j] - current_row[i])
                    return False
                
                start_val = current_row[-1] + 1
                end_val = M
                
                # Optimization: If pos is the last one, we can just find the first valid x.
                # But backtracking handles it.
                
                for x in range(start_val, end_val + 1):
                    new_diffs = []
                    valid = True
                    for y in current_row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                        
                    # Check internal uniqueness (always true for distinct y)
                    # Check against existing diffs in current_row
                    existing = set()
                    for i in range(len(current_row) - 1):
                        for j in range(i + 1, len(current_row) - 1):
                            existing.add(current_row[j] - current_row[i])
                            
                    if set(new_diffs) & existing:
                        continue
                        
                    current_row.append(x)
                    if fill_row(current_row, pos + 1):
                        return True
                    current_row.pop()
                
                return False
            
            if fill_row([0], 1):
                return True
            return False
        
        if backtrack(0):
            return rows
        return None

    # Try to improve
    for M in range(lower_bound, greedy_max):
        if time.time() > deadline - 5:
            break
        sol = solve_with_max(M)
        if sol:
            return sol
            
    return greedy_rows
