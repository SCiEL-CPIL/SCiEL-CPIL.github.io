"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

def construct(n_rows, k):
    """
    Randomized greedy set-packing with backtracking.
    
    Pre-compute valid Sidon-like row configurations (sets of k+1 increasing integers
    starting at 0) up to the counting bound, sort by max entry, and iteratively select
    rows with minimal scope that don't conflict with previously selected rows.
    Uses random tie-breaking and limited backtracking to escape local dead-ends.
    """
    # Counting bound for m(n_rows, k): we need n_rows rows, each with C(k+1, 2) differences
    # Total distinct differences needed: n_rows * C(k+1, 2)
    # If max entry is M, differences range from 1 to M, so M >= n_rows * C(k+1, 2)
    num_diffs_per_row = (k + 1) * k // 2
    counting_bound = n_rows * num_diffs_per_row
    
    # We'll search up to a reasonable upper bound
    # Start from counting_bound and go up if needed
    upper_limit = max(counting_bound + 20, 120)
    
    # Pre-compute all valid row configurations up to upper_limit
    # A row is a list of k+1 increasing integers starting at 0
    # It's valid if all C(k+1,2) differences within the row are distinct
    valid_rows = []
    
    # Generate all possible rows of length k+1 starting at 0, ending at most upper_limit
    # This could be large, so we need to be smart
    # For small k and n_rows, this is feasible
    
    def generate_rows(limit):
        """Generate all valid rows (Sidon sets) of size k+1 starting at 0, max <= limit."""
        rows = []
        
        def backtrack(current, start):
            if len(current) == k + 1:
                # Check if all differences are distinct
                diffs = set()
                valid = True
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        d = current[j] - current[i]
                        if d in diffs:
                            valid = False
                            break
                        diffs.add(d)
                    if not valid:
                        break
                if valid:
                    rows.append(list(current))
                return
            
            # Next element must be > last element and <= limit
            last = current[-1] if current else -1
            # Optimization: if we need k+1-len(current) more elements, and each must be at least 1 apart
            # The minimum possible next value is last + 1
            # The maximum possible next value is limit - (k + 1 - len(current) - 1)
            # because we need room for the remaining elements
            
            remaining = k + 1 - len(current)
            min_next = last + 1
            max_next = limit - (remaining - 1)
            
            if min_next > max_next:
                return
            
            for x in range(min_next, max_next + 1):
                # Quick check: new differences with existing elements must be distinct
                new_diffs = set()
                conflict = False
                for a in current:
                    d = x - a
                    # Check against existing diffs in current
                    for b in current:
                        if b != a:
                            existing_d = abs(b - a)
                            # Actually we need to check all pairs, but for efficiency:
                            # Just add to new_diffs and check at the end
                    new_diffs.add(d)
                
                # Check new_diffs against all existing differences in current
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        existing_d = current[j] - current[i]
                        if existing_d in new_diffs:
                            conflict = True
                            break
                    if conflict:
                        break
                
                if conflict:
                    continue
                
                # Also check new_diffs for internal conflicts (shouldn't happen since x > all in current)
                # new_diffs are all x - a for a in current, and since current is strictly increasing,
                # these are all distinct automatically
                
                current.append(x)
                backtrack(current, x)
                current.pop()
        
        backtrack([0], 0)
        return rows
    
    # Generate valid rows
    valid_rows = generate_rows(upper_limit)
    
    # Sort by maximum entry (scope)
    valid_rows.sort(key=lambda r: r[-1])
    
    # Now use randomized greedy with backtracking to select n_rows non-conflicting rows
    # A conflict occurs if two rows share any difference value
    
    def row_diffs(row):
        """Get the set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Pre-compute differences for all valid rows
    row_diff_sets = [row_diffs(r) for r in valid_rows]
    
    best_solution = None
    best_scope = float('inf')
    
    # Try multiple random seeds for tie-breaking
    for attempt in range(50):
        random.seed(attempt)
        
        selected_rows = []
        used_diffs = set()
        
        def backtrack_select(idx, selected, used):
            nonlocal best_solution, best_scope
            
            if len(selected) == n_rows:
                scope = max(r[-1] for r in selected)
                if scope < best_scope:
                    best_scope = scope
                    best_solution = list(selected)
                return True
            
            # Pruning: if we can't fill remaining rows with available rows, return
            remaining_needed = n_rows - len(selected)
            if idx >= len(valid_rows):
                return False
            
            # Try selecting from valid_rows starting at idx
            # Randomize the order slightly for tie-breaking
            candidates = list(range(idx, len(valid_rows)))
            
            # Sort candidates by scope, then randomize ties
            def sort_key(i):
                return (valid_rows[i][-1], random.random())
            
            candidates.sort(key=sort_key)
            
            for i in candidates:
                if len(selected) + (len(valid_rows) - i) < n_rows:
                    break  # Not enough candidates left
                
                # Check if this row conflicts with used differences
                if row_diff_sets[i] & used:
                    continue
                
                # Select this row
                new_used = used | row_diff_sets[i]
                new_selected = selected + [valid_rows[i]]
                
                if backtrack_select(i + 1, new_selected, new_used):
                    # We found a solution; in a full search we'd continue, but for time limits
                    # we can return after finding a good one
                    if best_solution is not None:
                        return True
                else:
                    continue
            
            return False
        
        # Run the backtracking selection
        backtrack_select(0, [], set())
        
        # If we found a solution, check if it's good enough
        if best_solution is not None:
            # Early exit if we hit the counting bound
            if best_scope <= counting_bound:
                break
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: use the original greedy approach if backtracking didn't find anything
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
    return rows

# EVOLVE-BLOCK-END
