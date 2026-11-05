import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0  # Leave 5s for overhead
    
    def greedy_solve():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows, max(max(r) for r in rows)

    initial_rows, best_score = greedy_solve()
    best_rows = [r[:] for r in initial_rows]
    
    # If the problem is very small, we can try to improve.
    # Heuristic: n_rows * k < 20 is very small.
    # For n=4, k=7, n*k=28.
    
    def backtrack(row_idx, current_row, used_diffs, current_max):
        nonlocal best_rows, best_score
        
        if time.time() - start_time > time_limit:
            return
        
        if row_idx == n_rows:
            if current_max < best_score:
                best_score = current_max
                best_rows = [r[:] for r in current_row]
            return
        
        # Prune if current max is already >= best
        if current_max >= best_score:
            return
            
        build_row(row_idx, current_row, used_diffs, current_max)
        
    def build_row(row_idx, row, used_diffs, current_max):
        nonlocal best_rows, best_score
        
        if time.time() - start_time > time_limit:
            return
            
        if len(row) == k + 1:
            # Row complete
            # Note: used_diffs already contains all diffs from this row
            # because we passed updated used_diffs in recursion.
            # Wait, in the recursion below, I will pass updated used_diffs.
            # So here, used_diffs is the set of ALL diffs from rows 0..row_idx.
            
            new_max = max(current_max, row[-1])
            if new_max < best_score:
                backtrack(row_idx + 1, row + [0], used_diffs, new_max) 
                # Wait, next row starts with 0.
                # But `backtrack` expects `current_row` to be the list of completed rows?
                # No, my `backtrack` signature was `backtrack(row_idx, current_row, ...)`.
                # This is confusing. Let's restructure.
                
            return

        # Try next mark
        start_x = row[-1] + 1
        # We can limit x to best_score - 1
        end_x = best_score - 1
        
        for x in range(start_x, end_x + 1):
            if time.time() - start_time > time_limit:
                return
            
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                continue
                
            # Recurse
            # New used_diffs includes new_diffs
            build_row(row_idx, row + [x], used_diffs | new_diffs, current_max)
            
            # If we found a better solution, best_score might have decreased.
            # We can break if x >= best_score
            if x >= best_score:
                break

    # The structure above is flawed because `backtrack` and `build_row` mix row index and mark index.
    # Let's use a single recursive function that handles both.
    
    def search(rows, current_row, used_diffs, current_max):
        nonlocal best_rows, best_score
        
        if time.time() - start_time > time_limit:
            return
            
        if len(current_row) == k + 1:
            # Finished a row
            new_rows = rows + [current_row]
            new_max = max(current_max, current_row[-1])
            
            if new_max >= best_score:
                return
                
            if len(new_rows) == n_rows:
                if new_max < best_score:
                    best_score = new_max
                    best_rows = [r[:] for r in new_rows]
            else:
                # Start next row
                search(new_rows, [0], used_diffs, new_max)
            return
        
        # Add next mark to current row
        start_x = current_row[-1] + 1
        end_x = best_score - 1
        
        for x in range(start_x, end_x + 1):
            if time.time() - start_time > time_limit:
                return
                
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
                
            search(rows, current_row + [x], used_diffs | new_diffs, current_max)
            
            if x >= best_score:
                break

    # Only run search if we have time and problem is small enough
    # n_rows * k < 50
    if n_rows * k < 50:
        search([], [0], set(), 0)

    return best_rows
