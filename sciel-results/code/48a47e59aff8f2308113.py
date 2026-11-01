"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    
    Uses backtracking with forward checking and dynamic variable ordering.
    """
    if k <= 0:
        return [[0] for _ in range(n_rows)]
    
    # Total number of differences per row: k*(k+1)//2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Simple counting bound: we need at least total_diffs distinct differences.
    # The maximum difference in a row is at most m (the largest entry).
    # So we need m >= total_diffs - 1 (differences are 1..m, so m values available)
    # But actually differences can be up to m, so we need m >= total_diffs.
    # Let's use this as a lower bound for the search space.
    
    # We'll search for the minimal m (largest entry) such that a valid configuration exists.
    # For each candidate m, we try to construct the rows using backtracking.
    
    # Lower bound on m: we need total_diffs distinct positive differences, each <= m.
    # So m must be at least total_diffs.
    lower_bound_m = total_diffs
    
    # Upper bound: use the greedy result as an initial upper bound
    def greedy_construct(n_rows, k):
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
        return rows, max(max(r) for r in rows)
    
    greedy_rows, greedy_max = greedy_construct(n_rows, k)
    upper_bound_m = greedy_max
    
    # If the greedy solution is already good, return it
    if lower_bound_m >= upper_bound_m:
        return greedy_rows
    
    # Try to find a better solution using backtracking
    # We'll search for m from lower_bound_m to upper_bound_m - 1
    
    def backtrack_construct(n_rows, k, max_val):
        """
        Try to construct n_rows rows with largest entry <= max_val.
        Returns the list of rows if successful, None otherwise.
        """
        # We'll build rows one by one. For each row, we choose k+1 values from {0, ..., max_val}
        # such that all differences within the row are distinct and not used by previous rows.
        
        # State: used_diffs (set of all differences used so far), rows constructed so far
        # For efficiency, we'll use a recursive approach with forward checking.
        
        used_diffs = set()
        rows = []
        
        def can_place_row(row_vals):
            """Check if a row with given values has all differences unused."""
            diffs = set()
            for i in range(len(row_vals)):
                for j in range(i + 1, len(row_vals)):
                    d = row_vals[j] - row_vals[i]
                    if d in used_diffs or d in diffs:
                        return False
                    diffs.add(d)
            return True
        
        def get_new_diffs(row_vals):
            """Get the set of differences for a row."""
            diffs = set()
            for i in range(len(row_vals)):
                for j in range(i + 1, len(row_vals)):
                    d = row_vals[j] - row_vals[i]
                    diffs.add(d)
            return diffs
        
        def construct_row(row_idx, current_row, start_val):
            """
            Recursively construct row row_idx.
            current_row: list of values chosen so far for this row (starts with [0])
            start_val: minimum value to try next (must be > last element)
            """
            if len(current_row) == k + 1:
                # Check if this row's differences are all valid
                diffs = get_new_diffs(current_row)
                if diffs & used_diffs:
                    return False
                # Place this row
                used_diffs.update(diffs)
                rows.append(current_row[:])
                
                if row_idx == n_rows - 1:
                    return True
                
                # Recurse to next row
                return construct_row(row_idx + 1, [0], 1)
            
            # Try next value
            last_val = current_row[-1]
            # The next value must be > last_val and <= max_val
            # Also, we need to ensure that we can still complete the row
            # with the remaining slots
            
            # Dynamic variable ordering: try values that lead to more constrained
            # situations first (fail early)
            
            # For the first value after 0, we have more flexibility
            # For later values, we're more constrained
            
            # Let's try values from last_val + 1 to max_val
            # But we can prune: the remaining (k+1 - len(current_row)) values must fit
            # in the range [next_val, max_val], so we need enough room
            
            remaining_slots = k + 1 - len(current_row)
            # The minimum span needed: if we have r slots left, the minimum additional
            # span is at least r (values must be strictly increasing by at least 1)
            # But also the differences must be distinct and not in used_diffs
            
            # Simple pruning: max_val - last_val >= remaining_slots
            if max_val - last_val < remaining_slots:
                return False
            
            # Try candidate values
            # To improve performance, we can prioritize values that create "rare" differences
            # But for now, let's try in order and rely on forward checking
            
            for next_val in range(last_val + 1, max_val + 1):
                # Forward checking: check if adding next_val creates any conflict
                # with already-chosen values in this row
                temp_row = current_row + [next_val]
                temp_diffs = set()
                valid = True
                for i in range(len(temp_row)):
                    for j in range(i + 1, len(temp_row)):
                        d = temp_row[j] - temp_row[i]
                        if d in used_diffs or d in temp_diffs:
                            valid = False
                            break
                        temp_diffs.add(d)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check if we can still complete the row
                # The remaining values must be > next_val and <= max_val
                # And their differences with all current values must be unused
                # This is complex, so we'll just recurse and let backtracking handle it
                
                result = construct_row(row_idx, temp_row, next_val)
                if result:
                    return True
                
                # Undo
                # used_diffs is modified only when a complete row is placed,
                # so no undo needed here for partial rows
                # But rows is modified only on success, so no undo needed
                
            return False
        
        # Start constructing row 0
        if construct_row(0, [0], 1):
            return rows
        return None
    
    # Try to find a better m
    best_rows = greedy_rows
    best_m = greedy_max
    
    # We'll try m values from lower_bound_m to upper_bound_m - 1
    # But this could be slow, so let's be strategic
    
    # Actually, let's try a few promising m values first
    # The counting bound is total_diffs, but the actual minimum might be higher
    
    # Let's try m = lower_bound_m, lower_bound_m + 1, etc. up to upper_bound_m - 1
    # But limit the search to avoid timeout
    
    start_time = time.time()
    timeout = 110  # seconds, leaving 10s buffer
    
    for m in range(lower_bound_m, upper_bound_m):
        if time.time() - start_time > timeout:
            break
        
        result = backtrack_construct(n_rows, k, m)
        if result is not None:
            best_rows = result
            best_m = m
            # Found a valid solution with this m, which is better than greedy
            # Since we're searching in increasing order, this is the best for this m
            # But we could potentially do even better, so continue searching
            # However, since m is increasing, the first valid m is the minimum
            break
    
    return best_rows

# EVOLVE-BLOCK-END
