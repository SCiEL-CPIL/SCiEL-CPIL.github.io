import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Calculate theoretical lower bound for scope
    # Total distinct differences = n_rows * (k+1)*k/2
    # These must be distinct positive integers <= scope
    # So scope >= n_rows * (k+1)*k/2
    lower_bound = n_rows * (k + 1) * k // 2
    
    # Phase 1: Generate an initial solution using a fast greedy method
    # This provides an upper bound for the backtracking search
    best_rows = _greedy_construct(n_rows, k)
    if not best_rows:
        # Fallback if greedy fails (should not happen)
        best_rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) <= k:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
                    x += 1
                else:
                    x += 1
            best_rows.append(row)
        
    best_scope = _scope(best_rows)
    
    # If we already hit the lower bound, we are done
    if best_scope == lower_bound:
        return best_rows

    # Phase 2: Backtracking search with symmetry breaking
    # State:
    # rows: list of completed rows
    # used_diffs: set of all differences used in completed rows and the current partial row
    rows = []
    used_diffs = set()
    timeout_flag = False
    
    def backtrack_safe(row_idx, current_row):
        nonlocal best_scope, best_rows, timeout_flag
        
        if timeout_flag:
            return
        
        if time.time() > deadline:
            timeout_flag = True
            return
        
        # If we have constructed all rows
        if row_idx == n_rows:
            scope = max(r[-1] for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
            return
        
        # Start new row with 0
        if not current_row:
            current_row = [0]
        
        # If the current row is complete
        if len(current_row) == k + 1:
            # Symmetry breaking: enforce non-decreasing scope of rows
            # This reduces the search space by n_rows!
            if row_idx > 0 and current_row[-1] < rows[row_idx - 1][-1]:
                return
            
            # Add row to completed list
            rows.append(current_row)
            
            # Recurse to next row
            backtrack_safe(row_idx + 1, [])
            
            # Backtrack: remove row
            rows.pop()
            
            # Remove this row's differences from used_diffs
            n = len(current_row)
            for i in range(n):
                for j in range(i + 1, n):
                    used_diffs.remove(current_row[j] - current_row[i])
            return
        
        # Try to extend the current row
        last_x = current_row[-1]
        
        # We only care about x < best_scope, as we want to improve
        for x in range(last_x + 1, best_scope):
            if timeout_flag:
                return
            
            # Check if x creates any conflicting differences
            new_diffs = set()
            valid = True
            for y in current_row:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Commit to x
            current_row.append(x)
            used_diffs.update(new_diffs)
            
            # Recurse
            backtrack_safe(row_idx, current_row)
            
            # Undo commit
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
            if timeout_flag:
                return
            if best_scope == lower_bound:
                return

    backtrack_safe(0, [])
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _greedy_construct(n_rows, k):
    """
    Fast greedy constructor to provide an initial upper bound.
    Picks the smallest valid integer for each position.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            x = last_x + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            used.update(new_diffs)
        rows.append(row)
    return rows
