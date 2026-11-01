"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of length k+1,
    starting at 0, strictly increasing, such that all positive differences
    within each row are distinct across all rows.
    
    Strategy: Use a backtracking search with iterative deepening on the scope.
    This approach is more systematic than simulated annealing and typically
    finds optimal or near-optimal solutions for small problem sizes much faster.
    """
    
    # The total number of differences per row is k*(k+1)/2
    # Total differences required: n_rows * k*(k+1)/2
    # These must be distinct positive integers.
    # The smallest possible set of distinct positive integers of size N is {1, ..., N}.
    # So the scope must be at least such that the largest difference fits.
    # Actually, the counting bound is roughly sqrt(2 * N).
    # Let's estimate a lower bound for the scope.
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # A very loose lower bound: the differences must be distinct.
    # The largest difference in a row of max element M is M.
    # If we have total_diffs distinct differences, they must be at least 1, 2, ..., total_diffs.
    # So the largest difference must be at least total_diffs.
    # Since the largest difference in any row is at most Scope, Scope >= total_diffs is not quite right.
    # The differences are subsets of {1, ..., Scope}.
    # We need total_diffs distinct values from {1, ..., Scope}.
    # So Scope >= total_diffs.
    # Wait, the differences are a_j - a_i. The max difference is a_k - a_0 = a_k.
    # So the max element is the max difference in that row.
    # So Scope >= total_diffs is a lower bound? No.
    # Example: k=2, n=1. Diffs: 2. Scope >= 2? Row [0, 1, 3]. Diffs 1, 2, 3. Scope 3.
    # Counting bound: We need distinct diffs.
    
    # Let's start the iterative deepening from a reasonable lower bound.
    # The "simple counting bound" mentioned in the prompt is 112 for (4,7).
    # Let's verify: k=7, diffs/row = 28. Total diffs = 112.
    # If all diffs are distinct, the set of diffs is a subset of {1, ..., Scope}.
    # So Scope must be at least 112.
    
    lower_bound = total_diffs
    
    # We will try scopes from lower_bound upwards.
    # To avoid searching too many scopes if the optimal is high, we can jump.
    # But for (4,7), 112 is the bound, best known is 134.
    # So we will search 112, 113, ...
    
    # To make the search faster, we can use a randomized backtracker that
    # tries to find *any* solution within a given scope.
    # If we find one, we return it.
    # We iterate scope from lower_bound up to a cap (e.g., 200).
    
    # We use a recursive backtracking function.
    # State:
    # - row_idx: which row we are constructing (0 to n_rows-1)
    # - current_row: the list of elements in the current row so far
    # - used_diffs: set of all differences used so far in all completed rows and current row
    
    # Optimization:
    # Pre-compute the required number of diffs for remaining rows?
    # Not strictly necessary for small sizes.
    
    # To handle the "minimize scope" efficiently, we iterate scope.
    # For each scope, we run the backtracker.
    # If the backtracker finds a solution, we return it.
    
    # To prevent infinite loops or timeouts, we add a node limit or time check.
    # Since we must finish in 120s, and this is small, it should be fine.
    
    # Randomization:
    # The order in which we try values can be randomized to escape local minima
    # or different structures, but for "smallest valid value" greedy backtracking,
    # it's deterministic. To improve robustness, we can randomize the order of
    # values slightly or the order of rows.
    
    # Let's stick to deterministic first, but allow random restarts if it fails
    # to find a solution within a scope (though if a solution exists, backtracking
    # should find it eventually, but it might be slow).
    
    # Better: Use a heuristic to order the values.
    # Try smaller values first to keep scope low.
    
    def solve_with_scope(scope):
        """
        Attempt to construct the set with all elements <= scope.
        Returns a list of rows if successful, else None.
        """
        rows = []
        used_diffs = set()
        
        # Heuristic: Construct rows in a specific order.
        # Sometimes constructing rows in a random order helps avoid specific dead ends
        # if the problem is symmetric, but here rows are just labels.
        # We'll just do 0 to n_rows-1.
        
        def backtrack(row_idx, current_row):
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row[:])
                # Recurse to next row
                if backtrack(row_idx + 1, [0]):
                    return True
                # If next row failed, remove this row and try next value
                rows.pop()
                return False
            
            # Determine the next element to add
            # It must be > current_row[-1] and <= scope
            start_val = current_row[-1] + 1
            end_val = scope
            
            # Heuristic: Try values from start_val to end_val
            # We can randomize this order to explore different parts of the space
            # if we suspect the greedy order is bad, but greedy usually works well
            # for minimizing scope.
            
            for val in range(start_val, end_val + 1):
                # Check if adding val creates any duplicate differences
                # Differences are val - x for x in current_row
                valid = True
                new_diffs = []
                for x in current_row:
                    d = val - x
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Add val and new diffs
                current_row.append(val)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if backtrack(row_idx, current_row):
                    return True
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        # Start with first row
        if backtrack(0, [0]):
            return rows
        return None

    # Iterative Deepening
    # We try scopes from lower_bound upwards.
    # To speed up, we can step by 1.
    # If we reach a timeout, we should return the best found so far.
    # But the task requires a valid result.
    # We assume a solution exists within a reasonable range.
    
    # To avoid getting stuck in a very slow branch for a specific scope,
    # we can add a node limit. If we exceed it, we try the next scope?
    # No, if we can't find a solution in scope S, we must try S+1.
    # But backtracking is complete. If a solution exists, it will find it.
    # The risk is time.
    
    # Let's add a simple time check.
    start_time = time.time()
    timeout = 100.0  # Leave some buffer for Python overhead
    
    # We'll try a few scopes.
    # If we fail to find a solution in a scope, we move to the next.
    # We can also randomize the search a bit by shuffling the value order
    # if the deterministic search is too slow, but for (4,7) it should be fast.
    
    # Actually, for (4,7), the greedy backtracker might be slow if it hits dead ends.
    # A common trick is to use a "most constrained variable" or similar.
    # But let's try the basic one first.
    
    # To improve performance, we can pre-calculate the differences for the current row
    # incrementally.
    
    # Let's implement a slightly smarter backtracker that keeps track of
    # the differences of the current row to avoid recalculating them.
    
    def solve_with_scope_smart(scope):
        rows = []
        used_diffs = set()
        
        def backtrack(row_idx, current_row, current_row_diffs):
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row[:])
                if backtrack(row_idx + 1, [0], set()):
                    return True
                rows.pop()
                return False
            
            start_val = current_row[-1] + 1
            end_val = scope
            
            # Try values
            for val in range(start_val, end_val + 1):
                # Check diffs
                valid = True
                new_diffs = []
                for x in current_row:
                    d = val - x
                    if d in used_diffs or d in current_row_diffs:
                        # d in current_row_diffs is redundant if we update it, 
                        # but we are checking against the set of diffs *in the current row* 
                        # that are already formed.
                        # Actually, current_row_diffs contains diffs between elements 
                        # already in current_row.
                        # The new diffs are val - x.
                        # We must check if any (val-x) is in used_diffs (global) 
                        # or if (val-x) conflicts with other new diffs?
                        # No, new diffs are distinct because x are distinct.
                        # But we must check if (val-x) is in current_row_diffs?
                        # Yes, because those diffs are already "used" in this row.
                        
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Also check if new_diffs conflict with each other?
                # No, they are val - x1, val - x2. Since x1 != x2, diffs are distinct.
                
                # Commit
                current_row.append(val)
                for d in new_diffs:
                    used_diffs.add(d)
                    current_row_diffs.add(d)
                
                if backtrack(row_idx, current_row, current_row_diffs):
                    return True
                
                # Undo
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
                    current_row_diffs.discard(d)
            
            return False
        
        if backtrack(0, [0], set()):
            return rows
        return None

    # Try scopes
    # We can parallelize or just run sequentially.
    # Given the small size, sequential is fine.
    
    # To handle potential slowness, we can add a node counter and reset it for each scope.
    # If we exceed a node limit (e.g., 100,000), we might assume no solution or just try next?
    # No, backtracking is exact. If we limit nodes, we might miss a solution.
    # But for (4,7), the search space is manageable.
    
    # Let's try from lower_bound to lower_bound + 50.
    # If not found, expand range.
    
    current_scope = lower_bound
    while current_scope < lower_bound + 100:
        if time.time() - start_time > timeout:
            break
        
        result = solve_with_scope_smart(current_scope)
        if result is not None:
            return result
        
        current_scope += 1
    
    # Fallback: If we didn't find a solution in the range, 
    # or ran out of time, we should return a valid (though maybe not optimal) set.
    # We can use a greedy construction as a fallback to ensure validity.
    # The greedy construction from the seed program is valid.
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    return greedy_construct()

# EVOLVE-BLOCK-END
