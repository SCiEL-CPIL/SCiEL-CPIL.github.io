import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    Uses a 'Generate and Select' strategy with backtracking.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Lower bound for the max value
    # Total differences needed: n_rows * k*(k+1)/2
    # Available differences in [1, M]: M
    # So M >= n_rows * k*(k+1)/2
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    
    def generate_candidate():
        """
        Generates a single valid row (list of k+1 integers starting at 0) 
        using randomized greedy.
        """
        used_diffs = set()
        row = [0]
        for _ in range(k):
            # Find next element
            # Randomize the search window
            # We want to keep values small, so search from low to high
            # But add some randomness to explore different structures
            
            # Determine a range to search
            # A good heuristic is to search up to current_max + some factor
            # Or just scan from row[-1] + 1
            x = row[-1] + 1
            found = False
            # Limit search depth for a single candidate generation to avoid hanging
            # If we can't find one quickly, it's a bad candidate
            max_search_steps = 500 
            steps = 0
            
            while not found:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    # Valid
                    # Add to row
                    row.append(x)
                    used_diffs.update(new_diffs)
                    found = True
                else:
                    x += 1
                    steps += 1
                    if steps > max_search_steps:
                        return None # Failed to generate a reasonable candidate
            
        return row

    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def can_select_n(rows, n):
        """
        Checks if we can select n non-conflicting rows from the given list.
        Uses backtracking.
        rows should be sorted by max value.
        """
        if len(rows) < n:
            return False
        
        # Sort by max value to prioritize smaller ones
        # (They are already sorted in the main loop if we pass filtered list)
        
        selected = []
        used_diffs = set()
        
        def backtrack(idx):
            if len(selected) == n:
                return True
            if idx >= len(rows):
                return False
            
            # Pruning: if we can't pick enough remaining rows, fail
            if len(rows) - idx < n - len(selected):
                return False
            
            # Option 1: Pick rows[idx]
            row = rows[idx]
            diffs = get_diffs(row)
            
            # Check conflict
            if not (diffs & used_diffs):
                selected.append(row)
                # Update used_diffs
                old_len = len(used_diffs)
                used_diffs.update(diffs)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                selected.pop()
                # Restore used_diffs (simple way: rebuild or subtract)
                # Since diffs are unique to this row in this branch, 
                # we can just remove them. But careful if diffs overlap with others?
                # No, we checked conflict, so diffs are disjoint from current used_diffs.
                # So we can safely remove diffs.
                used_diffs.difference_update(diffs)
                
            # Option 2: Skip rows[idx]
            if backtrack(idx + 1):
                return True
                
            return False
            
        return backtrack(0)

    # Main Loop: Binary Search on M (max value)
    
    # 1. Generate a pool of candidates
    # We need a pool large enough to likely contain a solution.
    # Let's generate a few hundred.
    
    pool = []
    num_candidates_to_generate = 1000
    gen_time_limit = 30 # seconds for generation
    
    while len(pool) < num_candidates_to_generate and time.time() - start_time < gen_time_limit:
        c = generate_candidate()
        if c is not None:
            # Check if it's "good" (small max)
            # We can filter out very large ones later
            pool.append(c)
            
    if not pool:
        # Fallback: return a simple greedy result if generation fails completely
        # (Should not happen)
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k+1:
                nd = {x-a for a in row}
                if not (nd & used):
                    row.append(x)
                    used.update(nd)
                x += 1
            rows.append(row)
        return rows

    # 2. Sort pool by max element
    pool.sort(key=lambda r: r[-1])
    
    # 3. Binary search for the minimum M
    # Low bound: max(max(r) for r in pool[:n]) if we could pick n smallest? 
    # Actually, just use the max of the n-th smallest element in the pool.
    if len(pool) < n_rows:
        # Not enough candidates, just return the best we have (might be invalid if < n_rows)
        # Pad with greedy if necessary? 
        # For safety, if we don't have enough, we return the pool as is (invalid if len < n_rows)
        # But we generated 1000, so len should be >= n_rows.
        pass
        
    low = pool[n_rows-1][-1] if len(pool) >= n_rows else pool[-1][-1]
    high = pool[-1][-1]
    
    best_solution = None
    
    while low <= high and time.time() - start_time < time_limit - 5:
        mid = (low + high) // 2
        
        # Filter pool to rows with max <= mid
        # Note: pool is sorted, so we can just take a prefix
        # Find the index where max > mid
        # Since sorted by max, we can binary search or just iterate
        # For efficiency, since we do this in a loop, let's precompute maxes
        # Or just use a list comprehension (fast for small N)
        
        candidates = [r for r in pool if r[-1] <= mid]
        
        if len(candidates) < n_rows:
            low = mid + 1
            continue
            
        # Try to select n_rows non-conflicting rows from candidates
        if can_select_n(candidates, n_rows):
            best_solution = candidates # We found a feasible set for mid
            high = mid - 1
        else:
            low = mid + 1
            
    if best_solution is not None:
        # We have a set of candidates that *contains* a valid selection.
        # We need to extract the specific n_rows.
        # Re-run the selection to get the actual rows.
        # We can just use the logic from can_select_n but capture the result.
        
        # Let's write a helper that returns the rows
        def find_selection(rows, n):
            selected = []
            used_diffs = set()
            
            def backtrack(idx):
                if len(selected) == n:
                    return True
                if idx >= len(rows):
                    return False
                if len(rows) - idx < n - len(selected):
                    return False
                
                row = rows[idx]
                diffs = get_diffs(row)
                
                if not (diffs & used_diffs):
                    selected.append(row)
                    used_diffs.update(diffs)
                    if backtrack(idx + 1):
                        return True
                    selected.pop()
                    used_diffs.difference_update(diffs)
                    
                if backtrack(idx + 1):
                    return True
                return False
                
            if backtrack(0):
                return selected
            return None

        final_rows = find_selection(best_solution, n_rows)
        if final_rows:
            return final_rows

    # Fallback: Return the n smallest rows from the pool (might have conflicts, but better than nothing? 
    # No, must be valid. 
    # If binary search failed to find a valid set in the pool, we need to generate more or use a different strategy.
    # Given the time, let's just return the result of a simple greedy if the above fails.
    
    # Simple Greedy Fallback
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k+1:
            nd = {x-a for a in row}
            if not (nd & used):
                row.append(x)
                used.update(nd)
            x += 1
        rows.append(row)
    return rows
