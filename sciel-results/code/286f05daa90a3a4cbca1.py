import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 120.0  # Keep a margin from the 150s limit
    
    # Total differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # First, get a valid greedy solution as fallback
    used_diffs_greedy = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            x = current_max + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs_greedy:
                    break
                x += 1
            row.append(x)
            current_max = x
            used_diffs_greedy |= new_diffs
        greedy_rows.append(row)
    
    greedy_scope = max(max(r) for r in greedy_rows)
    best_rows = [r[:] for r in greedy_rows]
    best_scope = greedy_scope
    
    # Backtracking with forward checking
    # We'll try to construct each row, choosing values that minimize the maximum
    # while ensuring all differences are unique across all rows.
    
    def backtrack_row(row_idx, current_row, used_diffs, max_val_so_far, depth):
        """
        Try to complete row row_idx, starting from current_row.
        Returns (completed_row, new_used_diffs, new_max) or None if failed.
        """
        if time.time() > deadline - 5.0:
            return None
        
        if len(current_row) == k + 1:
            return (current_row[:], used_diffs, max_val_so_far)
        
        prev = current_row[-1]
        # Try values from prev+1 upward, but bound by some heuristic
        # We want to keep the scope small, so try smaller values first
        # but we need to ensure we can complete the row
        
        # Heuristic: the remaining elements need at least (k+1 - len(current_row)) - 1 more steps
        # Each step needs at least +1, so minimum final value is prev + (k+1 - len(current_row))
        # But we also need to avoid used differences
        
        min_possible_final = prev + (k + 1 - len(current_row))
        
        # Try candidate values
        # Start from prev+1, but we might want to skip ahead if conflicts are high
        # For now, try sequentially with some randomization for diversity
        
        candidates = []
        # Generate candidates: from prev+1 to some upper bound
        # Upper bound: we don't want to exceed best_scope if possible
        upper = max(prev + 50, min_possible_final + 20)  # Reasonable upper bound
        upper = min(upper, best_scope + 10)  # Don't go too far beyond current best
        
        for x in range(prev + 1, upper + 1):
            if time.time() > deadline - 5.0:
                return None
            
            # Check if adding x creates any conflict with used_diffs
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            
            # Forward checking: can we still complete the row?
            # The next element must be > x, and we need (k+1 - len(current_row) - 1) more elements
            # Each needs to create new differences
            # Simple check: there should be enough "space" 
            # This is a weak check, but helps prune
            
            # Try this candidate
            new_row = current_row + [x]
            new_used = used_diffs | new_diffs
            new_max = max(max_val_so_far, x)
            
            # If we're at the last position, we're done
            if len(new_row) == k + 1:
                return (new_row, new_used, new_max)
            
            # Recurse
            result = backtrack_row(row_idx, new_row, new_used, new_max, depth + 1)
            if result is not None:
                return result
        
        return None
    
    # Try to construct all rows using backtracking
    # We'll process rows in order, and for each row, try to find a valid completion
    
    # Strategy: Try multiple random orderings of which values to prefer
    # and use random restarts if stuck
    
    num_restarts = 50
    best_found = best_scope
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        used_diffs = set()
        current_rows = []
        success = True
        
        for row_idx in range(n_rows):
            if time.time() > deadline - 5.0:
                success = False
                break
            
            # Try to construct this row
            # Start with [0]
            row = [0]
            
            # Use backtracking to complete the row
            result = backtrack_row(row_idx, row, used_diffs, 0, 0)
            
            if result is None:
                # Backtracking failed for this row
                success = False
                break
            
            completed_row, used_diffs, row_max = result
            current_rows.append(completed_row)
        
        if success:
            scope = max(max(r) for r in current_rows)
            if scope < best_found:
                best_found = scope
                best_rows = [r[:] for r in current_rows]
                best_scope = scope
    
    # If backtracking didn't improve, return the greedy solution
    if best_found >= greedy_scope:
        return greedy_rows
    
    return best_rows
