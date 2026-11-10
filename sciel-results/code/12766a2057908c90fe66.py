import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118  # 2s safety margin for 120s limit
    
    # Total differences needed
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Theoretical lower bound based on counting: m(m+1)/2 >= total_diffs
    # m^2 + m - 2*total_diffs >= 0
    # m >= (-1 + sqrt(1 + 8*total_diffs)) / 2
    lower_bound = int((-1 + (1 + 8 * total_diffs) ** 0.5) / 2)
    
    best_solution = None
    best_score = float('inf')
    
    def backtrack(rows, used_diffs, row_idx, col_idx, current_row, max_val_so_far, time_limit):
        nonlocal best_solution, best_score
        
        if time.time() > time_limit:
            return False, max_val_so_far
            
        if row_idx == n_rows:
            # All rows constructed
            score = max(max(row) for row in rows)
            if score < best_score:
                best_score = score
                best_solution = [r[:] for r in rows]
            return True, max_val_so_far
            
        if col_idx == k + 1:
            # Current row complete, move to next
            return backtrack(rows, used_diffs, row_idx + 1, 0, [], max_val_so_far, time_limit)
            
        # Determine search range for current element
        # If col_idx == 0, element is 0
        if col_idx == 0:
            candidates = [0]
        else:
            last_val = current_row[-1]
            # Heuristic: don't search too far ahead if we are already worse than best
            # Also, prune if current potential max exceeds best_score
            # The smallest possible max for this row is last_val + (k - col_idx)
            # But we need to ensure we can fit the remaining elements.
            
            # Simple pruning: if last_val + (k - col_idx) >= best_score, 
            # it's unlikely to improve, but we might still find a valid solution 
            # if best_score is inf or if we are in a restart that might find a better one.
            # However, if we are already at a value >= best_score, any extension will be >= best_score.
            # So if last_val >= best_score, we can prune this branch.
            if last_val >= best_score:
                return False, max_val_so_far
            
            # Start from last_val + 1
            # Upper bound for search: 
            # 1. best_score (if we want to improve)
            # 2. Some heuristic upper bound to avoid infinite loops in a single step
            # Let's search up to min(best_score, last_val + 50 + random_noise)
            # Actually, for backtracking to work well, we need to explore valid options.
            # If we restrict too tightly, we might miss solutions.
            # But if we search too wide, it's slow.
            
            # Strategy: 
            # 1. Try values in increasing order.
            # 2. Stop if x > best_score (since we want to minimize max).
            # 3. If best_score is inf, we need a cap. Let's use a generous cap initially.
            
            search_limit = best_score if best_score < float('inf') else (last_val + 100 + int(total_diffs**0.5))
            
            candidates = []
            for x in range(last_val + 1, search_limit + 1):
                # Check if x creates new differences that are unused
                # New diffs: x - a for a in current_row
                # If any of these is in used_diffs, x is invalid.
                
                # Optimization: Check only the largest difference? No, all must be distinct.
                # But if x - current_row[0] is in used_diffs, it's invalid.
                # Actually, if any x - a is in used_diffs, it's invalid.
                
                # Quick check: 
                # The differences are x, x-current_row[0], ..., x-current_row[-1]
                # Wait, current_row[0] is 0. So x - 0 = x.
                # If x is in used_diffs, invalid.
                if x in used_diffs:
                    continue
                    
                # Check other differences
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
                    
            # If no candidates found, backtrack (fail)
            if not candidates:
                return False, max_val_so_far
            
            # Randomize order of candidates? 
            # For minimization, smaller x is generally better for the current row,
            # but might block future rows.
            # Let's shuffle the candidates to explore different branches.
            # But bias towards smaller values?
            # Let's try: sort by value, then pick randomly from the first few?
            # Or just shuffle all? Shuffling all might pick a large x early, leading to high max.
            # Better: Sort by value. Then, with some probability, pick a random one from the valid list.
            # Or: Iterate through sorted candidates, but with random restarts, we will try different paths.
            
            # To make backtracking effective, we should try the most promising (smallest) first,
            # but if it fails, we backtrack.
            # However, if we always try smallest first, it's deterministic unless we randomize the "failure" threshold.
            # Let's shuffle the candidates. This ensures we explore different structures.
            random.shuffle(candidates)
            
        # Try candidates
        for x in candidates:
            new_row = current_row + [x]
            new_diffs = set()
            for a in current_row:
                new_diffs.add(x - a)
            
            # Check if new_diffs overlap with used_diffs (should be handled in candidate gen, but double check)
            # Actually, candidate gen already checked.
            
            new_used = used_diffs | new_diffs
            new_max = max(max_val_so_far, x)
            
            # Recursive step
            success, _ = backtrack(rows, new_used, row_idx, col_idx + 1, new_row, new_max, time_limit)
            
            if success:
                # If we found a solution, we can return immediately if we are in a "find any" mode,
                # but we want the BEST. So we continue searching?
                # No, backtracking usually finds the first solution in the search order.
                # If we shuffled, the first solution might not be optimal.
                # To find the optimal, we would need to explore all branches, which is too slow.
                # Instead, we rely on multiple restarts with different shuffles.
                # So, if we find a valid solution, we update best and return True to stop this branch.
                return True, new_max
            
            # If this candidate failed, try the next one
            # But we must be careful with time.
            
        # If all candidates failed
        return False, max_val_so_far

    # Initial deterministic greedy to get a baseline
    def greedy_deterministic():
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
        score = max(max(row) for row in rows)
        return score, rows

    # Run deterministic first
    try:
        det_score, det_rows = greedy_deterministic()
        best_score = det_score
        best_solution = det_rows
    except Exception:
        pass

    # Now run randomized backtracking restarts
    # Each restart gets a small time budget
    # Total time available: deadline - time.time()
    
    # We will run restarts until time runs out
    # Each restart:
    # 1. Reset state
    # 2. Run backtracking with a time limit for that restart
    # 3. If it finds a solution better than best, update
    
    # How to allocate time?
    # If we give each restart 1 second, we get ~118 restarts.
    # If we give each restart 10ms, we get many more, but they might not go deep.
    # Backtracking needs depth. Let's try a few longer restarts first, then shorter?
    # Or just fixed short restarts.
    
    # Let's try a hybrid:
    # 1. A few "deep" searches with larger time budgets (e.g., 5s each) to find good solutions.
    # 2. Many "shallow" searches with small budgets (e.g., 0.1s each) to explore variety.
    
    # Actually, since we have a global best, we can prune heavily.
    # Let's just run restarts with a fixed small time slice, say 0.5s.
    
    restart_time_slice = 0.5
    seed = 42
    
    while time.time() < deadline - restart_time_slice:
        random.seed(seed)
        seed += 1
        
        # For each restart, we can also randomize the order of rows?
        # The rows are indistinguishable, so order doesn't matter for the final set.
        # But the construction order matters for the "used" set.
        # Backtracking constructs row 0, then row 1, etc.
        
        # We can also randomize the "shuffle" inside the backtracking.
        # The `random.shuffle(candidates)` inside backtracking uses the global random state.
        # So different seeds will lead to different shuffles.
        
        # Call backtracking
        # We pass a time limit for this specific restart
        restart_deadline = time.time() + restart_time_slice
        
        # We need to reset the "success" flag or similar?
        # The backtracking function updates global best_solution.
        # It returns True if it found a valid solution (not necessarily optimal, just valid).
        # We want to find the MINIMUM.
        # So if it finds a solution, we update best_score.
        # Then we stop that restart (return True).
        
        # But what if it finds a solution that is WORSE than current best?
        # The backtracking function checks `if score < best_score`.
        # So it only updates if it's better.
        # But it still returns True if it finds ANY valid solution.
        # This is fine, as we stop the restart once a valid solution is found.
        # To find a BETTER solution, we need to explore deeper or different branches.
        # The shuffling helps explore different branches.
        
        # One issue: if the first valid solution found in a restart is bad, we stop.
        # We might miss a better one in the same restart.
        # To mitigate this, we could modify backtracking to continue searching for a better solution
        # within the time limit. But that's complex.
        # Simpler: Just rely on many restarts.
        
        # Let's run the backtracking
        # We need to be careful with the `time_limit` parameter.
        # It checks `time.time() > time_limit`.
        
        rows = []
        used_diffs = set()
        # Call backtrack
        # We don't care about the return value of max_val, just the side effect on best_solution
        backtrack(rows, used_diffs, 0, 0, [], 0, restart_deadline)
        
        # Check if we improved
        # best_score is updated inside backtrack
        
    return best_solution if best_solution is not None else []

