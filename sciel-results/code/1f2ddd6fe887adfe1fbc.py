"""Difference triangle set (7,5) solver using optimized backtracking with iterative deepening."""
import sys
import time
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    
    Uses iterative deepening backtracking with symmetry breaking and pruning.
    """
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    # For (7, 5): 7 rows, 6 elements, 15 diffs/row, 105 total diffs.
    # Counting bound: 105. Best known: 112.
    
    # We search for the minimal M such that a valid configuration exists.
    # We use iterative deepening: try M = lower_bound, lower_bound+1, ...
    
    # Lower bound: total_diffs (since diffs are 1..M and distinct)
    # Actually, the diffs are distinct integers. If max entry is M, max diff is M.
    # We need total_diffs distinct positive integers.
    # So M must be at least total_diffs.
    # However, the structure of the rows imposes stronger constraints.
    # For (7,5), total_diffs = 105.
    
    lower_bound = total_diffs
    upper_bound = 200  # Safe upper bound for this small case
    
    def solve(max_m):
        """
        Try to construct n_rows rows with all elements <= max_m.
        Returns the list of rows if successful, None otherwise.
        """
        used_diffs = set()
        rows = []
        
        # Symmetry breaking: The first row can be fixed to a specific pattern 
        # or we can just ensure rows are sorted by their largest element?
        # A stronger symmetry break: Fix the first row to be [0, 1, ...] is not possible 
        # as it uses small diffs. 
        # Instead, we can enforce that the largest element of row i is <= largest element of row i+1?
        # No, that doesn't reduce symmetry much.
        
        # Better symmetry break: 
        # The set of differences is what matters. 
        # We can enforce that the first row is the one with the smallest maximum element?
        # Or simply, since rows are indistinguishable in the final set, 
        # we can enforce an ordering on the rows based on their largest element.
        # Let's enforce: max(row[i]) <= max(row[i+1]).
        
        def backtrack(row_idx, current_row, used_diffs):
            if row_idx == n_rows:
                return rows[:]
            
            # Pruning: 
            # The remaining rows (row_idx to n_rows-1) must have max elements >= current_row's max if we enforce ordering.
            # But we are building rows one by one.
            
            # If we are building row `row_idx`, we must ensure its max element <= max_m.
            # Also, if we enforce non-decreasing max elements, 
            # then max(current_row) must be >= max(rows[row_idx-1]) if row_idx > 0.
            
            min_start = 0
            if row_idx > 0:
                # To break symmetry, enforce max(row[i]) <= max(row[i+1])
                # This means the current row's max must be at least the previous row's max.
                # Wait, if we enforce non-decreasing max, we are just ordering the rows.
                # This reduces the search space by n_rows!
                min_start = rows[row_idx-1][-1]
            else:
                min_start = 0

            # We need to find a valid row for this index.
            # We iterate through possible max values for this row?
            # No, we build the row element by element.
            
            # To speed up, we can try to construct the row with the smallest possible max element
            # that is >= min_start and <= max_m.
            
            # However, a simple backtracking on elements is better.
            
            # Let's use a helper to find one valid row given constraints.
            # But we need to explore different rows to satisfy future constraints.
            
            # Standard backtracking:
            # Try to add elements to current_row.
            
            if len(current_row) == k + 1:
                max_val = current_row[-1]
                if max_val < min_start:
                    return None
                if max_val > max_m:
                    return None
                
                # Check if this row is valid (diffs unused)
                # This is checked during construction.
                
                rows.append(current_row)
                
                # Update used_diffs
                new_used = set(used_diffs)
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        new_used.add(current_row[j] - current_row[i])
                
                result = backtrack(row_idx + 1, [0], new_used)
                if result is not None:
                    return result
                
                rows.pop()
                return None
            
            # Determine next element
            last = current_row[-1]
            # Next element x must be > last
            # And x <= max_m
            # And x - a_i not in used_diffs for any a_i in current_row
            
            # Pruning: We need to fit k+1-len(current_row) more elements.
            # The smallest possible next elements are last+1, last+2, ...
            # So the minimum possible max for this row is last + (k+1 - len(current_row))
            # This must be <= max_m
            # And if row_idx > 0, this min max must be >= min_start?
            # Actually, if the min possible max < min_start, we can just start from a higher value?
            # No, the elements must be increasing.
            
            # Let's iterate x from last+1 to max_m
            # But we can prune if even with smallest steps we can't reach a valid state?
            
            # Heuristic: Try smaller x first? Or larger?
            # Smaller x uses smaller diffs, which are more "conflicting" potentially?
            # Usually, packing problems benefit from using small numbers early.
            
            # Optimization: 
            # If we are at the last element (len(current_row) == k), 
            # we just need to find a valid x.
            
            # Range for x:
            start_x = last + 1
            end_x = max_m
            
            # Additional pruning: 
            # The remaining elements (count = k+1 - len(current_row)) must fit in [x, max_m].
            # So x + (remaining - 1) <= max_m
            # x <= max_m - (remaining - 1)
            remaining = k + 1 - len(current_row)
            if remaining > 1:
                end_x = min(end_x, max_m - (remaining - 1))
            
            for x in range(start_x, end_x + 1):
                # Check diffs
                valid = True
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Recursive step
                current_row.append(x)
                result = backtrack(row_idx, current_row, used_diffs | new_diffs)
                if result is not None:
                    return result
                current_row.pop()
            
            return None
        
        return backtrack(0, [0], set())

    # Iterative Deepening
    # Start from lower_bound
    # To handle the time limit, we can add a timeout or node limit
    # But for (7,5), the solution is known to be 112.
    # Lower bound 105.
    # We will try M from 105 upwards.
    
    # To speed up, we can use a random seed or multiple attempts if the first fails?
    # No, backtracking is deterministic.
    
    # The search space might be large. 
    # Let's add a simple node counter to prevent infinite loops if M is too small, 
    # though the recursion depth is bounded.
    
    nodes_visited = 0
    max_nodes = 500000 # Safety limit
    
    def backtrack_limited(row_idx, current_row, used_diffs):
        nonlocal nodes_visited
        nodes_visited += 1
        if nodes_visited > max_nodes:
            return None
        
        if row_idx == n_rows:
            return rows[:]
        
        min_start = 0
        if row_idx > 0:
            min_start = rows[row_idx-1][-1]
        
        if len(current_row) == k + 1:
            max_val = current_row[-1]
            if max_val < min_start or max_val > max_m:
                return None
            
            rows.append(current_row)
            new_used = set(used_diffs)
            for i in range(len(current_row)):
                for j in range(i+1, len(current_row)):
                    new_used.add(current_row[j] - current_row[i])
            
            result = backtrack_limited(row_idx + 1, [0], new_used)
            if result is not None:
                return result
            
            rows.pop()
            return None
        
        last = current_row[-1]
        start_x = last + 1
        end_x = max_m
        
        remaining = k + 1 - len(current_row)
        if remaining > 1:
            end_x = min(end_x, max_m - (remaining - 1))
        
        # Optimization: If we have already used many small diffs, 
        # maybe we should jump?
        
        for x in range(start_x, end_x + 1):
            valid = True
            new_diffs = set()
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            current_row.append(x)
            result = backtrack_limited(row_idx, current_row, used_diffs | new_diffs)
            if result is not None:
                return result
            current_row.pop()
        
        return None

    # We need to wrap the solve function to use the limited version
    # But the `rows` and `max_m` are local to `solve`.
    # Let's refactor to put the logic inside the loop.

    for M in range(lower_bound, upper_bound + 1):
        rows = []
        used_diffs = set()
        nodes_visited = 0
        
        def bt(row_idx, current_row):
            nonlocal nodes_visited, used_diffs
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return False
            
            if row_idx == n_rows:
                return True
            
            min_start = 0
            if row_idx > 0:
                min_start = rows[row_idx-1][-1]
            
            if len(current_row) == k + 1:
                max_val = current_row[-1]
                if max_val < min_start or max_val > M:
                    return False
                
                # Check if all diffs are unique globally
                # This is already ensured by the 'valid' check in the loop below?
                # No, the 'valid' check only checks against `used_diffs` passed down.
                # We need to update `used_diffs` before recursing.
                
                new_used = set(used_diffs)
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        new_used.add(current_row[j] - current_row[i])
                
                rows.append(current_row)
                old_used = used_diffs
                used_diffs = new_used
                
                if bt(row_idx + 1, [0]):
                    return True
                
                used_diffs = old_used
                rows.pop()
                return False
            
            last = current_row[-1]
            start_x = last + 1
            end_x = M
            
            remaining = k + 1 - len(current_row)
            if remaining > 1:
                end_x = min(end_x, M - (remaining - 1))
            
            # Heuristic: Try values that don't conflict
            # We can also try to order x by some cost?
            # For now, linear scan.
            
            for x in range(start_x, end_x + 1):
                valid = True
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                current_row.append(x)
                if bt(row_idx, current_row):
                    return True
                current_row.pop()
            
            return False

        if bt(0, [0]):
            return rows
    
    # If no solution found within upper_bound, return a greedy fallback
    # This should not happen for (7,5) with M=200
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
