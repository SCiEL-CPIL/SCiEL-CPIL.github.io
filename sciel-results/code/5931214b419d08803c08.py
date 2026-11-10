"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n_rows, k):
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def greedy_row(used):
        """Build one row greedily, returning the row and updated used set."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        return row, used
    
    def try_construction(order):
        """Try constructing rows in the given order. Returns (rows, max_val) or None."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row, used = greedy_row(used)
            rows.append(row)
        max_val = max(max(r) for r in rows)
        return rows, max_val
    
    # Strategy 1: Standard greedy (current approach)
    rows, best_max = try_construction(range(n_rows))
    
    # Strategy 2: Try different orderings by varying the greedy choices
    # Use backtracking with limited depth to find better solutions
    def dfs_construct(row_idx, current_rows, used, depth_limit=3):
        """DFS with limited backtracking to try alternative choices."""
        if time.time() > deadline:
            return None, None
        
        if row_idx == n_rows:
            max_val = max(max(r) for r in current_rows)
            return list(current_rows), max_val
        
        # Try greedy first
        row_greedy = [0]
        x = 0
        while len(row_greedy) < k + 1:
            x += 1
            new_diffs = {x - a for a in row_greedy}
            if not new_diffs & used:
                row_greedy.append(x)
                used_greedy = used | new_diffs
                break
            # If we can't place this x, continue
            else:
                used_greedy = used
        
        # Actually, let's do proper DFS: try each possible x that works
        best_result = None
        best_max = float('inf')
        
        row = [0]
        x = 0
        found_any = False
        
        # Try to build row with backtracking on last few positions
        def build_row_with_backtrack(pos, current_row, current_used):
            nonlocal best_result, best_max, found_any
            if time.time() > deadline:
                return False
            
            if pos == k:
                # Complete row
                new_row = current_row[:]
                max_val = max(new_row)
                if max_val < best_max:
                    best_max = max_val
                    best_result = list(current_rows) + [new_row]
                    found_any = True
                return True
            
            # Try each possible x starting from current_row[-1] + 1
            for x in range(current_row[-1] + 1, current_row[-1] + 50):
                new_diffs = {x - a for a in current_row}
                if not new_diffs & current_used:
                    if build_row_with_backtrack(pos + 1, current_row + [x], current_used | new_diffs):
                        if found_any and pos >= k - depth_limit:
                            return True  # Stop after finding good solutions near the end
            
            return found_any
        
        # This is getting complex; let's use a simpler approach:
        # Try multiple random restarts with slight variations
        
        return best_result, best_max
    
    # Simpler: try several random perturbations of the greedy choice
    import random
    random.seed(42)
    
    best_rows = rows
    best_max_val = best_max
    
    attempts = 0
    while time.time() < deadline - 10 and attempts < 50:
        attempts += 1
        used = set()
        trial_rows = []
        success = True
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Occasionally skip a valid x to explore different space
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    if random.random() < 0.1 and x < 100:
                        continue  # Skip this valid x sometimes
                    row.append(x)
                    used |= new_diffs
            trial_rows.append(row)
        
        max_val = max(max(r) for r in trial_rows)
        if max_val < best_max_val:
            best_max_val = max_val
            best_rows = trial_rows
    
    return best_rows
# EVOLVE-BLOCK-END
