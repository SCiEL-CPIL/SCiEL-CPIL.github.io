"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses a backtracking search with constraint propagation and iterative deepening.
    """
    start_time = time.time()
    timeout = 110  # seconds, leave margin for evaluator overhead
    
    # Total number of differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on maximum entry
    lower_bound = total_diffs  # counting bound
    
    # Precompute all possible difference pairs for a row of length k+1
    # For a row [0, a1, a2, ..., ak], the differences are:
    # a1, a2, ..., ak, a2-a1, a3-a1, ..., ak-a1, a3-a2, ..., ak-a2, ..., ak-a(k-1)
    
    def get_row_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid_row(row, used_diffs):
        """Check if a row's differences are all not in used_diffs."""
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs:
                    return False
        return True
    
    def backtrack(rows, used_diffs, current_max, deadline):
        """
        Backtracking search to find n_rows valid rows.
        Returns (success, rows_list, max_entry)
        """
        if time.time() > deadline:
            return False, rows, current_max
        
        if len(rows) == n_rows:
            return True, rows, current_max
        
        # Try to build the next row
        row = [0]
        
        def build_row(pos, used_diffs_local, current_max_local):
            nonlocal row
            if time.time() > deadline:
                return False, used_diffs_local, current_max_local
            
            if pos == k + 1:
                # Row complete, check validity
                if not is_valid_row(row, used_diffs_local):
                    return False, used_diffs_local, current_max_local
                
                new_used = used_diffs_local | get_row_diffs(row)
                new_max = max(current_max_local, row[-1])
                
                rows.append(list(row))
                success, rows_after, final_max = backtrack(rows, new_used, new_max, deadline)
                if success:
                    return True, rows_after, final_max
                rows.pop()
                return False, used_diffs_local, current_max_local
            
            last = row[-1]
            # Try candidate values
            # Use a heuristic: try smaller values first to keep max low
            # But also need to ensure enough room for remaining elements
            
            # Minimum value for next element
            min_next = last + 1
            # Maximum value: we need room for (k+1 - pos - 1) more elements after this
            # Each subsequent element must be at least 1 more than the previous
            # So max possible for this element depends on how much we want to keep max low
            
            # Try a range of values
            # Heuristic: try values that leave enough room
            remaining_positions = k - pos
            # We need to place remaining_positions more elements, each at least 1 apart
            # So the last element must be at least last + 1 + remaining_positions
            # To keep max low, we should try values that allow compact placement
            
            # Try values in a smart order
            candidates = []
            # First try compact values
            for x in range(min_next, min_next + 30):
                # Check if x - a_i is not in used_diffs_local for all a_i in row
                valid = True
                for a in row:
                    if (x - a) in used_diffs_local:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
            
            # Also try some larger values for diversity
            for x in range(min_next + 30, min_next + 60, 5):
                valid = True
                for a in row:
                    if (x - a) in used_diffs_local:
                        valid = False
                        break
                if valid and x not in candidates:
                    candidates.append(x)
            
            for x in candidates:
                new_row = row + [x]
                new_diffs = {x - a for a in row}
                
                # Check if new_diffs conflict with used_diffs_local
                if new_diffs & used_diffs_local:
                    continue
                
                new_used_local = used_diffs_local | new_diffs
                new_max_local = max(current_max_local, x)
                
                # Recursively build the rest of the row
                row = new_row
                success, used_after, max_after = build_row(pos + 1, new_used_local, new_max_local)
                if success:
                    return True, used_after, max_after
                row = new_row[:-1]
            
            return False, used_diffs_local, current_max_local
        
        success, _, max_after = build_row(1, used_diffs, current_max)
        if success:
            return True, rows, max_after
        
        return False, rows, current_max
    
    # Try to find a solution with maximum entry as small as possible
    # Use iterative deepening: try max_entry = lower_bound, lower_bound+1, ...
    
    best_solution = None
    best_max = float('inf')
    
    # First, try to find any solution with a reasonable max
    # Then try to improve it
    
    # Strategy: 
    # 1. Try greedy to get an initial solution
    # 2. Use backtracking to try to improve
    
    # Greedy approach
    def greedy():
        used = set()
        rows = []
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
        return rows
    
    # Get initial greedy solution
    greedy_rows = greedy()
    greedy_max = max(row[-1] for row in greedy_rows)
    best_solution = greedy_rows
    best_max = greedy_max
    
    # Now try to improve using backtracking
    # Try to find a solution with max < best_max
    
    # Use backtracking with a time limit
    deadline = start_time + timeout - 5  # leave 5 seconds for safety
    
    # Try backtracking with different max bounds
    for target_max in range(lower_bound, best_max):
        if time.time() > deadline:
            break
        
        # Try to find a solution with max entry <= target_max
        # We can add this as a constraint in the backtracking
        
        def backtrack_limited(rows, used_diffs, current_max, deadline, max_bound):
            if time.time() > deadline:
                return False, rows, current_max
            
            if len(rows) == n_rows:
                if current_max <= max_bound:
                    return True, rows, current_max
                return False, rows, current_max
            
            row = [0]
            
            def build_row_limited(pos, used_diffs_local, current_max_local):
                nonlocal row
                if time.time() > deadline:
                    return False, used_diffs_local, current_max_local
                
                if current_max_local > max_bound:
                    return False, used_diffs_local, current_max_local
                
                if pos == k + 1:
                    if not is_valid_row(row, used_diffs_local):
                        return False, used_diffs_local, current_max_local
                    
                    new_used = used_diffs_local | get_row_diffs(row)
                    new_max = max(current_max_local, row[-1])
                    
                    if new_max > max_bound:
                        return False, used_diffs_local, current_max_local
                    
                    rows.append(list(row))
                    success, rows_after, final_max = backtrack_limited(rows, new_used, new_max, deadline, max_bound)
                    if success:
                        return True, rows_after, final_max
                    rows.pop()
                    return False, used_diffs_local, current_max_local
                
                last = row[-1]
                min_next = last + 1
                
                # Limit candidates to stay within max_bound
                max_candidate = min(min_next + 50, max_bound)
                
                for x in range(min_next, max_candidate + 1):
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs_local:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_row = row + [x]
                    new_diffs = {x - a for a in row}
                    
                    if new_diffs & used_diffs_local:
                        continue
                    
                    new_used_local = used_diffs_local | new_diffs
                    new_max_local = max(current_max_local, x)
                    
                    if new_max_local > max_bound:
                        continue
                    
                    row = new_row
                    success, used_after, max_after = build_row_limited(pos + 1, new_used_local, new_max_local)
                    if success:
                        return True, used_after, max_after
                    row = new_row[:-1]
                
                return False, used_diffs_local, current_max_local
            
            success, _, max_after = build_row_limited(1, used_diffs, current_max)
            if success:
                return True, rows, max_after
            
            return False, rows, current_max
        
        success, solution_rows, max_found = backtrack_limited([], set(), 0, deadline, target_max)
        if success:
            best_solution = solution_rows
            best_max = max_found
            break
    
    # If we found a better solution, return it
    if best_solution is not None:
        return best_solution
    
    # Fallback: return greedy solution
    return greedy_rows

# EVOLVE-BLOCK-END
