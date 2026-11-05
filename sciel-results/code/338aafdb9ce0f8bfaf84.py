import sys
import time

def construct(n_rows, k):
    # n_rows=4, k=7 (row length 8)
    row_len = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * total_diffs_per_row
    
    # The minimum possible scope is total_diffs (if diffs are 1..total_diffs)
    # But they don't have to be contiguous.
    # Lower bound on scope:
    # The largest difference in a row of length L is at least L(L-1)/2 ? No.
    # In a Golomb ruler of order n, the length is at least n(n-1)/2.
    # Here we have multiple rulers.
    
    # We will search for the minimum scope.
    # Start from a lower bound.
    # A simple lower bound: The total number of distinct differences is N = n_rows * k(k+1)/2.
    # The differences are integers from 1 to M.
    # So M >= N.
    # N = 4 * 28 = 112.
    
    best_solution = None
    start_time = time.time()
    time_limit = 100 # Leave buffer for 120s limit
    
    # We'll try scopes from 112 upwards.
    # To be efficient, we might jump or use a heuristic.
    # But 112 to 134 is only 23 iterations.
    
    # Optimization: Pre-calculate all pairs (i, j) for a row of length row_len?
    # No, the values change.
    
    def backtrack(row_idx, current_row, used_diffs, max_scope_limit):
        """
        row_idx: which row we are constructing (0 to n_rows-1)
        current_row: list of ints in the current row
        used_diffs: set of all differences used so far
        max_scope_limit: the maximum allowed value for any element
        """
        if time.time() - start_time > time_limit:
            return None
            
        if row_idx == n_rows:
            return [row for row in [current_row]] # Placeholder, need to pass list of rows
            
        # This structure is better:
        # pass a list of completed rows
        pass

    # Let's restructure the state to be cleaner.
    def search(rows_completed, current_row, used_diffs, scope_limit):
        if time.time() - start_time > time_limit:
            return None
            
        if len(rows_completed) == n_rows - 1 and len(current_row) == row_len:
            # Check if current_row is valid (differences distinct and not in used_diffs)
            # This is handled by construction in the recursion step
            return rows_completed + [current_row]
            
        if len(current_row) == row_len:
            # Move to next row
            return search(rows_completed + [current_row], [0], used_diffs, scope_limit)
            
        # Extend current_row
        last_val = current_row[-1]
        # The next value must be > last_val and <= scope_limit
        # Also, we need to ensure that the differences formed with previous elements are not in used_diffs
        
        # Heuristic: Try values close to last_val first? Or spread out?
        # For Golomb rulers, spreading out is often better to keep differences large and distinct.
        # But we want to minimize the max value.
        
        # Let's iterate next_val from last_val + 1 to scope_limit
        for next_val in range(last_val + 1, scope_limit + 1):
            # Check differences
            valid = True
            new_diffs = []
            for prev_val in current_row:
                d = next_val - prev_val
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Also check if new_diffs have duplicates among themselves?
            # No, current_row is strictly increasing, so next_val - prev_val are distinct for different prev_val.
            # (next_val - a) != (next_val - b) if a != b.
            
            # Check symmetry breaking?
            # If this is not the last row, we should ensure that the max of this row 
            # is <= max of subsequent rows?
            # We don't know subsequent rows yet.
            # However, we can enforce that the rows are generated in order of increasing max value.
            # If rows_completed is not empty, let prev_max = max(rows_completed[-1])
            # We don't know the max of current_row yet, only the last element.
            # But the last element IS the max of the row (since it's increasing).
            # So if rows_completed is not empty:
            if rows_completed:
                prev_max = rows_completed[-1][-1]
                # The final max of current_row will be >= next_val.
                # We want max(current_row) >= prev_max ?
                # Actually, to break symmetry, we assume Row 0 <= Row 1 <= ... in terms of max element.
                # So we require next_val >= prev_max?
                # No, we require the FINAL max of current_row >= prev_max.
                # Since we are building up, if next_val < prev_max, it's possible the final max is still >= prev_max.
                # But if we are at the last element (len(current_row) == row_len - 1), then next_val IS the max.
                # So if len(current_row) == row_len - 1 and next_val < prev_max:
                #     continue
                
                # A stronger constraint:
                # If we are filling the last element of the row, it must be >= prev_max.
                if len(current_row) == row_len - 1:
                    if next_val < prev_max:
                        continue
                    # Also, to ensure strict ordering or just non-decreasing?
                    # Non-decreasing is enough for symmetry breaking.
                    # If two rows have the same max, they are distinct sets, but swapping them yields same solution.
                    # So allowing equal max is fine, but we must ensure we don't double count.
                    # Standard trick: enforce max(Row_i) <= max(Row_{i+1}).
                    # If max(Row_i) == max(Row_{i+1}), we might still have duplicates if the rows are identical.
                    # But the differences are distinct globally, so rows cannot be identical (unless n_rows=1).
                    # So non-decreasing max is a valid symmetry break.
            
            # Recurse
            new_used = used_diffs.union(new_diffs)
            result = search(rows_completed, current_row + [next_val], new_used, scope_limit)
            if result is not None:
                return result
                
        return None

    # Initialize
    # To make the search faster, we can fix the first row to be a "greedy" ruler?
    # No, that might miss the optimal.
    # But we can sort the rows by their max value.
    
    # Let's try scopes from 112 to 140
    for scope in range(112, 141):
        if time.time() - start_time > time_limit:
            break
        # print(f"Trying scope {scope}")
        res = search([], [0], set(), scope)
        if res is not None:
            return res
            
    # If no solution found in the range, return a greedy solution as fallback
    # Greedy construction
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
