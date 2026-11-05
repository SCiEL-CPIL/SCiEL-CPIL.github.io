import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy:
    1. Column-wise List Scheduling:
       View the construction as filling positions 1..k for each row.
       At each step, for each row, determine the set of "forbidden" differences
       that would be created if we pick a certain value.
       Pick the value that minimizes the "cost" (e.g., smallest value that doesn't
       conflict, or value that leaves the most room).
       This is a greedy heuristic that respects global constraints early.
       
    2. Conflict-Driven Local Search:
       If the greedy construction fails or yields a large scope, use a local search
       that identifies specific duplicate differences and attempts to fix them by
       moving entire rows or swapping elements between rows.
       
    3. Iterative Deepening on Scope:
       Try to construct a solution with scope M = lower_bound, then M+1, etc.
       Use the column-wise heuristic as the constructor.
    """
    
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin from 150s limit
    
    num_elements = k + 1
    num_pairs_per_row = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs_per_row
    
    # Lower bound
    lower_bound = total_diffs_needed
    
    # Helper: Calculate all differences for a row (list of sorted ints)
    def get_row_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    # Helper: Check if a set of new diffs conflicts with used diffs
    def has_conflict(new_diffs, used_diffs):
        for d in new_diffs:
            if d in used_diffs:
                return True
        return False

    # Helper: Get max element in rows
    def get_max_element(rows):
        return max(max(row) for row in rows) if rows else 0

    # Helper: Verify full validity
    def is_valid(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_row_diffs(r))
        return len(all_diffs) == len(set(all_diffs))

    # Column-wise List Scheduling Constructor
    def column_wise_construct(M, rng):
        """
        Try to construct rows with max element <= M.
        Returns list of rows or None if failed.
        """
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        # We fill positions 1 to k for each row
        # At each step, we decide the next value for each row.
        # To make it "list scheduling", we process rows in order of "urgency"
        # e.g., rows with the smallest current max element, or random order.
        
        # Actually, a simpler effective heuristic for this specific problem:
        # For each row, pick the smallest available value that doesn't conflict.
        # But to allow for different structures, we can randomize the order of rows
        # or the choice of value slightly.
        
        # Let's try a greedy approach where we fill each row completely one by one,
        # but with a "lookahead" or "repair" if stuck.
        # However, the inspiration suggests "column-wise" meaning we might fill
        # position 1 for all rows, then position 2 for all rows, etc.
        # This ensures that the differences involving the first element (which are
        # just the values themselves) are distinct globally early on.
        
        # Let's implement:
        # For pos in 1..k:
        #   For each row r (in some order):
        #     Find the smallest x > rows[r][-1] such that:
        #       1. x <= M
        #       2. For all y in rows[r], (x - y) not in used_diffs
        #     If no such x, fail (or backtrack/repair)
        #     Add x to rows[r], update used_diffs
        
        # To improve success rate, we can randomize the row order at each column.
        
        for pos in range(1, k + 1):
            # Shuffle row order to break symmetry and allow different structures
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            
            success = True
            for r in row_order:
                row = rows[r]
                last_val = row[-1]
                found = False
                
                # Try values from last_val + 1 to M
                # To avoid bias, we can start from a random point or use a heuristic
                # Let's try the smallest valid value first (greedy)
                # But sometimes picking a slightly larger value helps future steps.
                # We'll try the smallest first. If it fails, we might need backtracking.
                # For now, let's just try the smallest.
                
                for x in range(last_val + 1, M + 1):
                    # Check if adding x creates any duplicate diffs
                    new_diffs = [x - y for y in row]
                    if has_conflict(new_diffs, used_diffs):
                        continue
                    
                    # Also check if x itself is in used_diffs?
                    # No, used_diffs contains differences between elements.
                    # The difference x - 0 is x. So x must not be in used_diffs.
                    # This is covered by new_diffs since 0 is in row.
                    
                    # Accept x
                    row.append(x)
                    used_diffs.update(new_diffs)
                    found = True
                    break
                
                if not found:
                    return None
        
        return rows

    # Local Search: Repair conflicts by moving elements
    def local_search_refine(rows, M_current, rng):
        """
        Given a valid or near-valid solution, try to reduce the scope.
        If the input is invalid, try to fix conflicts.
        """
        current_max = get_max_element(rows)
        
        # If invalid, first try to make it valid by moving conflicting elements
        # But for simplicity, let's assume we start with a valid solution from greedy
        # and just try to reduce the max.
        
        # Identify the row with the max element
        max_val = current_max
        candidate_rows = [i for i, r in enumerate(rows) if max_val in r]
        
        if not candidate_rows:
            return rows, current_max
        
        # Try to reduce the max element in one of these rows
        # We pick a random row with the max element
        r_idx = rng.choice(candidate_rows)
        row = rows[r_idx]
        idx = row.index(max_val)
        
        # Try to replace max_val with a smaller value
        # New value must be > row[idx-1] and < row[idx+1] (if exists)
        lower_bound_val = row[idx-1] + 1 if idx > 0 else 1
        upper_bound_val = row[idx+1] - 1 if idx < len(row)-1 else max_val - 1
        
        if lower_bound_val > upper_bound_val:
            return rows, current_max
        
        # Try random values in range
        tried = set()
        for _ in range(20): # Limit attempts per step
            if lower_bound_val > upper_bound_val:
                break
            new_val = rng.randint(lower_bound_val, upper_bound_val)
            if new_val in tried:
                continue
            tried.add(new_val)
            
            # Apply
            old_val = row[idx]
            row[idx] = new_val
            row.sort()
            
            # Check validity
            if is_valid(rows):
                # Valid! Update max
                current_max = get_max_element(rows)
                return rows, current_max
            else:
                # Revert
                row[idx] = old_val
                row.sort()
        
        return rows, current_max

    # Main Logic
    best_solution = None
    best_M = float('inf')
    
    rng = random.Random(42)
    
    # Try M from lower_bound upwards
    # Since column-wise is fast, we can try many M values
    # But we need to be careful with time.
    
    # First, try to find a valid solution with a reasonably large M to ensure we have a fallback
    M_fallback = lower_bound + 30
    rows_fb = column_wise_construct(M_fallback, rng)
    if rows_fb and is_valid(rows_fb):
        best_solution = rows_fb
        best_M = get_max_element(rows_fb)
    
    # Now try to find a better solution with smaller M
    # We'll iterate M from lower_bound to best_M - 1
    # For each M, run column-wise construction multiple times with different random seeds
    # If a valid solution is found, update best and break (or continue to try smaller M? No, M is increasing)
    # Actually, we want the SMALLEST M. So we should try M = lower_bound, lower_bound+1, ...
    # If we find a valid solution for M, that's our best possible for that M.
    # We can stop if we find one for M = lower_bound.
    
    # However, column-wise might fail for small M even if a solution exists.
    # So we might need to use local search to repair failures or find solutions that greedy misses.
    
    # Let's try a hybrid:
    # For M in range(lower_bound, best_M):
    #   Try column-wise construct.
    #   If success and valid, update best and break.
    #   If fail, try a few random restarts.
    #   If still fail, maybe use local search on a "near-miss" solution?
    #     This is complex. Let's stick to greedy for now, but with multiple restarts.
    
    for M in range(lower_bound, best_M):
        if time.time() > DEADLINE - 10:
            break
        
        found = False
        # Try a few random restarts for this M
        for restart in range(5):
            rng_re = random.Random(42 + restart * 1000 + M)
            rows_try = column_wise_construct(M, rng_re)
            if rows_try and is_valid(rows_try):
                best_solution = rows_try
                best_M = M
                found = True
                break
        
        if found:
            # We found a solution for M. Since we're iterating upwards, this is the best.
            # We can break if M == lower_bound, otherwise continue?
            # No, if we found a solution for M, we don't need to check larger M.
            # But we started from lower_bound, so if we found one at M, it's the minimum found.
            # We can break immediately.
            break
    
    # If we didn't find anything better than the fallback, return the fallback
    if best_solution is None:
        # Emergency fallback: very large M
        M_emergency = lower_bound + 100
        rows_em = column_wise_construct(M_emergency, rng)
        if rows_em and is_valid(rows_em):
            best_solution = rows_em
        else:
            # Last resort: brute force greedy with no randomization
            rows_em = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                for _ in range(k):
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
                    row.append(x)
                    used.update(new_diffs)
                rows_em.append(row)
            best_solution = rows_em

    return best_solution
# EVOLVE-BLOCK-END
