import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # 1. Initial Greedy
    rows = []
    used = set()
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
    
    current_scope = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    # 2. Local Search
    # We will try to reduce the scope.
    # State: rows, used, scope
    
    def get_scope(rows):
        return max(max(r) for r in rows)
        
    def get_sum(rows):
        return sum(sum(r) for r in rows)
        
    # Helper to rebuild a row with a specific max limit
    def rebuild_row(idx, max_limit, used_others):
        # used_others: set of diffs used by other rows
        # We need to find a row of length k+1, starting 0, max <= max_limit
        # such that its diffs are disjoint from used_others.
        
        # Backtracking
        row = [0]
        
        def backtrack(current_row, current_max):
            if len(current_row) == k + 1:
                return current_row[:]
            
            # Heuristic: Try smaller x first? Or random?
            # To minimize max, we should try to keep x small.
            # But we must ensure we can complete the row.
            
            # Determine range for next x
            # Min next x: current_row[-1] + 1
            # Max next x: max_limit
            
            start_x = current_row[-1] + 1
            end_x = max_limit
            
            # Optional: Prune if not enough space
            # Min diff needed for remaining steps
            
            # Randomize order?
            # For finding a solution, greedy (small to large) is good for tight bounds.
            # For diversity, random.
            
            # Let's try small to large first (greedy)
            for x in range(start_x, end_x + 1):
                new_diffs = {x - a for a in current_row}
                if not new_diffs & used_others:
                    # Check if new_diffs have internal collisions? 
                    # No, current_row is strictly increasing, so diffs are distinct within the row.
                    # We only need to check against used_others.
                    
                    # Additional check: Can we still complete the row?
                    # The remaining marks must be > x.
                    # The smallest possible new diffs will be at least 1.
                    # This check is complex. Just try.
                    
                    res = backtrack(current_row + [x], x)
                    if res is not None:
                        return res
            return None
        
        return backtrack(row, 0)

    # Main Loop
    while time.time() < deadline:
        # Identify candidate row to improve
        # Prefer the row with the largest mark
        max_row_idx = max(range(n_rows), key=lambda i: max(rows[i]))
        
        # Try to reduce the scope
        # Target scope: current_scope - 1
        target_scope = current_scope - 1
        
        # If target_scope is too small (below counting bound?), skip
        # Counting bound for one row is not 105. 
        # The global bound is 105.
        
        # Remove current row's diffs
        current_diffs = set()
        for i in range(len(rows[max_row_idx])):
            for j in range(i+1, len(rows[max_row_idx])):
                current_diffs.add(rows[max_row_idx][j] - rows[max_row_idx][i])
        
        used_others = used - current_diffs
        
        # Try to rebuild with target_scope
        new_row = rebuild_row(max_row_idx, target_scope, used_others)
        
        if new_row:
            # Accept
            rows[max_row_idx] = new_row
            used = used_others | {new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i+1, len(new_row))}
            current_scope = max(max(r) for r in rows)
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        else:
            # Failed to reduce scope. 
            # Try to improve the row with same scope but lower sum?
            # Or try a random row?
            # Let's try to rebuild the same row with current_scope to see if we find a "better" configuration
            # (e.g. one that leaves more room for others).
            # This is heuristic.
            
            # To avoid infinite loops, we can just break or randomize.
            # Let's try a random row and see if we can improve its internal sum.
            if random.random() < 0.1:
                 # Random restart of a random row
                 idx = random.randint(0, n_rows-1)
                 # ... similar logic ...
                 pass

    return best_rows
