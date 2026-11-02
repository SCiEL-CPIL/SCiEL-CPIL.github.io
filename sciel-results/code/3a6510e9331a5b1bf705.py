import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set (Golomb rulers) with n_rows rows,
    each having k+1 marks. Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    # Initial greedy solution to set a baseline
    used_greedy = set()
    rows_greedy = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_greedy:
                row.append(x)
                used_greedy |= new_diffs
        rows_greedy.append(row)
    
    best_rows = rows_greedy
    best_max_entry = max(max(r) for r in rows_greedy)
    
    # If k is small or n_rows is small, DFS might be fast.
    # Total diffs = n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # DFS with backtracking
    used = set()
    current_rows = []
    
    def dfs(row_idx, current_row, current_max):
        nonlocal best_rows, best_max_entry, used, current_rows
        
        if time.time() - start_time > time_limit:
            return False # Signal timeout
        
        if row_idx == n_rows:
            if current_max < best_max_entry:
                best_max_entry = current_max
                best_rows = [list(r) for r in current_rows]
            return True
        
        if current_max >= best_max_entry:
            return True
        
        if len(current_row) == k + 1:
            # Move to next row
            current_rows.append(current_row)
            if dfs(row_idx + 1, [0], current_max):
                current_rows.pop()
                return True
            current_rows.pop()
            return False
        
        # Determine next mark
        last_mark = current_row[-1]
        # Lower bound: last_mark + 1
        # Upper bound: best_max_entry - 1 (since we want to improve)
        # Also, we need to ensure we can fit the remaining (k - len(current_row)) marks.
        # The smallest possible increment for each remaining mark is 1.
        # So the absolute minimum final mark is last_mark + (k - len(current_row) + 1)
        # But we also have difference constraints.
        
        # Heuristic: Try smaller x first to keep current_max low.
        # Range of x:
        min_x = last_mark + 1
        # We can't exceed best_max_entry
        max_x = best_max_entry - 1
        
        # Additional pruning:
        # If we pick x, the new differences are x - r for r in current_row.
        # The largest new difference is x - 0 = x.
        # So x must be < best_max_entry.
        
        # Optimization: If the current row is getting too long and the max is too high, prune.
        
        for x in range(min_x, max_x + 1):
            new_diffs = set()
            conflict = False
            for r in current_row:
                d = x - r
                if d in used:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
            
            # Check if we can even fit the remaining marks in this row
            # given the current max constraint?
            # The remaining marks must be > x.
            # The smallest possible next mark is x+1.
            # The final mark will be at least x + (k - len(current_row))
            # If x + (k - len(current_row)) >= best_max_entry, we might still be ok if we are in the last row?
            # No, if we are in the last row, the final mark is the max.
            # If x + (k - len(current_row)) >= best_max_entry, it means even with minimal increments,
            # we won't beat the current best.
            # Wait, we are trying to beat best_max_entry.
            # So the final mark must be < best_max_entry.
            # Final mark >= x + (k - len(current_row))
            # So if x + (k - len(current_row)) >= best_max_entry, we can prune this branch.
            if x + (k - len(current_row)) >= best_max_entry:
                continue
            
            used.update(new_diffs)
            current_row.append(x)
            
            # Recurse
            if dfs(row_idx, current_row, max(current_max, x)):
                current_row.pop()
                used -= new_diffs
                return True
            
            current_row.pop()
            used -= new_diffs
            
            # If timeout, break out
            if time.time() - start_time > time_limit:
                return False
                
        return False
        
    # Start DFS
    # We use [0] as the initial row
    dfs(0, [0], 0)
    
    return best_rows
