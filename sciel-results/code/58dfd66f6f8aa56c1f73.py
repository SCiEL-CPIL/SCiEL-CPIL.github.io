import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    L = k + 1
    diffs_per_row = L * (L - 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    for M in range(lower_bound, lower_bound + 100):
        # Step 1: Generate all valid first rows (Golomb rulers) with max <= M
        first_rows = _generate_golomb_rulers(L, M)
        
        # Step 2: For each first row, try to build the rest
        for first_row in first_rows:
            used_diffs = set(first_row[i] - first_row[j] for i in range(L) for j in range(i))
            
            # Check if remaining rows can be built
            remaining_rows = _build_remaining_rows(n_rows - 1, L, M, used_diffs)
            
            if remaining_rows is not None:
                return [first_row] + remaining_rows
                
    return _greedy_fallback(n_rows, k)

def _generate_golomb_rulers(L, M):
    """Generate all Golomb rulers of length L with max value <= M."""
    rulers = []
    
    def dfs(row):
        if len(row) == L:
            rulers.append(row[:])
            return
            
        last = row[-1]
        # Pruning: min max value for remaining L-len(row) elements
        # If we have r elements, we need L-r more.
        # Min next is last+1.
        # Min max is last + (L - len(row))
        if last + (L - len(row)) > M:
            return
            
        current_diffs = set(row[i] - row[j] for i in range(len(row)) for j in range(i))
        
        for x in range(last + 1, M + 1):
            # Check new differences
            valid = True
            for a in row:
                d = x - a
                if d in current_diffs:
                    valid = False
                    break
            if valid:
                row.append(x)
                dfs(row)
                row.pop()
                
    dfs([0])
    return rulers

def _build_remaining_rows(n_rem, L, M, used_diffs):
    """
    Build n_rem rows of length L, starting at 0, max <= M,
    using differences not in used_diffs.
    Returns list of rows or None.
    """
    if n_rem == 0:
        return []
        
    rows = []
    
    def dfs(row_idx, current_row):
        if row_idx == n_rem:
            return True
            
        if len(current_row) == L:
            rows.append(current_row)
            if dfs(row_idx + 1, [0]):
                return True
            rows.pop()
            return False
            
        last_val = current_row[-1]
        r = len(current_row)
        
        # Pruning: available diffs
        diffs_so_far_in_row = r * (r - 1) // 2
        diffs_remaining_in_row = (L * (L - 1) // 2) - diffs_so_far_in_row
        total_needed = diffs_remaining_in_row + (n_rem - row_idx - 1) * (L * (L - 1) // 2)
        
        if len(used_diffs) + total_needed > M:
            return False
            
        max_candidate = M - (L - r - 1)
        
        for x in range(last_val + 1, max_candidate + 1):
            new_diffs = set()
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            current_row.append(x)
            used_diffs.update(new_diffs)
            
            if dfs(row_idx, current_row):
                return True
                
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False
        
    if dfs(0, [0]):
        return rows
    return None

def _greedy_fallback(n_rows, k):
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
