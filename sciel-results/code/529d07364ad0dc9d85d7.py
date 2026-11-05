import time
import sys

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    
    Strategy: Backtracking search with heuristic ordering and pruning.
    We build rows one by one. For each row, we build elements one by one.
    At each step, we consider valid candidates, order them by a heuristic,
    and recurse. If we hit a dead end, we backtrack.
    """
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0
    
    best_result = None
    best_scope = float('inf')
    
    # Heuristic parameters
    # We penalize using small differences because they are scarce.
    # We also prefer smaller values to keep the scope low.
    
    def get_heuristic_score(x, row, used_diffs, alpha=1.0):
        """
        Score a candidate x. Lower is better.
        Primary: x (smaller is better)
        Secondary: Penalty for using small differences.
        """
        new_diffs = {x - a for a in row}
        # Count how many small differences are introduced
        small_penalty = sum(1 for d in new_diffs if d <= 10)
        return x + alpha * small_penalty

    def backtrack(rows, used_diffs, current_row_idx, current_row, current_scope_bound):
        """
        Recursive backtracking to build rows.
        returns: (success, best_scope_found, best_rows)
        """
        nonlocal best_result, best_scope, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False, float('inf'), None
            
        # If we've built all rows
        if current_row_idx == n_rows:
            # Calculate final scope
            final_scope = max(r[-1] for r in rows)
            if final_scope < best_scope:
                best_scope = final_scope
                best_result = [r[:] for r in rows]
                if best_scope <= 105:
                    return True, best_scope, best_result
            return False, best_scope, best_result
            
        # Start a new row
        if not current_row:
            current_row = [0]
            
        # If current row is complete
        if len(current_row) == k + 1:
            # Commit this row
            rows.append(current_row)
            success, scope, res = backtrack(rows, used_diffs, current_row_idx + 1, [], current_scope_bound)
            if success:
                return True, scope, res
            # Backtrack: remove the row
            rows.pop()
            return False, best_scope, best_result
            
        # Need to add next element to current_row
        last_val = current_row[-1]
        
        # Generate candidates
        # Upper bound is current best scope (or a heuristic bound if inf)
        upper_bound = best_scope - 1 if best_scope < float('inf') else 200
        
        candidates = []
        for x in range(last_val + 1, upper_bound + 1):
            new_diffs = {x - a for a in current_row}
            if not new_diffs & used_diffs:
                # Check if adding this x keeps us within the scope bound
                # The max value in the row so far will be x.
                # We need to ensure we can finish the row with values <= best_scope.
                # A simple check: x < best_scope
                if x < best_scope:
                    score = get_heuristic_score(x, current_row, used_diffs)
                    candidates.append((score, x, new_diffs))
                    
        if not candidates:
            return False, best_scope, best_result
            
        # Sort candidates by score
        candidates.sort(key=lambda item: item[0])
        
        # Limit branching factor to avoid explosion
        # Try top 20 candidates
        max_branches = 20
        for score, x, new_diffs in candidates[:max_branches]:
            # Try this candidate
            new_row = current_row + [x]
            new_used = used_diffs | new_diffs
            
            # Recurse
            success, scope, res = backtrack(rows, new_used, current_row_idx, new_row, current_scope_bound)
            if success:
                return True, scope, res
                
        return False, best_scope, best_result

    # Initial call
    # We might need multiple restarts if the first heuristic path fails or gets stuck in a local minimum.
    # Since the directive says "design a new program... from scratch", and "do not reuse current designs",
    # I will implement a loop that varies the heuristic parameter 'alpha' or uses a slightly randomized tie-breaking
    # to explore different parts of the search space, but the core is the backtracking CSP solver.
    
    # To ensure we find a good solution, we'll run the backtracking search multiple times
    # with slightly different randomizations in the tie-breaking of candidates with similar scores.
    
    import random
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Reset state
        best_result_current = None
        best_scope_current = float('inf')
        
        # We modify the global best_scope to allow pruning, but we need to be careful.
        # Actually, the inner function uses the outer 'best_scope'.
        # Let's wrap the backtracking in a way that it can be restarted with a fresh 'best_scope' 
        # or we just let it improve the global best.
        
        # To make it a "new design", I'll structure it as a class-like closure or just a clean recursive function
        # that takes the current state.
        
        # Let's try a single deep search first. If it finds a solution, great.
        # If not, we might need to restart with a different seed for tie-breaking.
        
        # To allow tie-breaking randomness, we can shuffle candidates with equal scores.
        
        def backtrack_with_random_tie(rows, used_diffs, current_row_idx, current_row):
            nonlocal best_result, best_scope, start_time, time_limit
            
            if time.time() - start_time > time_limit:
                return False
                
            if current_row_idx == n_rows:
                final_scope = max(r[-1] for r in rows)
                if final_scope < best_scope:
                    best_scope = final_scope
                    best_result = [r[:] for r in rows]
                    return True
                return False
                
            if not current_row:
                current_row = [0]
                
            if len(current_row) == k + 1:
                rows.append(current_row)
                success = backtrack_with_random_tie(rows, used_diffs, current_row_idx + 1, [])
                if success:
                    return True
                rows.pop()
                return False
                
            last_val = current_row[-1]
            upper_bound = best_scope - 1 if best_scope < float('inf') else 200
            
            candidates = []
            for x in range(last_val + 1, upper_bound + 1):
                new_diffs = {x - a for a in current_row}
                if not new_diffs & used_diffs:
                    if x < best_scope:
                        # Heuristic: Minimize x, then minimize small diffs
                        small_penalty = sum(1 for d in new_diffs if d <= 10)
                        # Add small random noise to break ties and explore different paths
                        noise = random.random() * 0.5
                        score = x + 1.0 * small_penalty + noise
                        candidates.append((score, x, new_diffs))
                        
            if not candidates:
                return False
                
            candidates.sort(key=lambda item: item[0])
            
            # Try top candidates
            # Increase branching as we go deeper? Or keep it fixed.
            max_branches = 10
            for score, x, new_diffs in candidates[:max_branches]:
                new_row = current_row + [x]
                new_used = used_diffs | new_diffs
                
                if backtrack_with_random_tie(rows, new_used, current_row_idx, new_row):
                    return True
                    
            return False

        # Run the search
        backtrack_with_random_tie([], set(), 0, [])
        
        if best_result is not None and best_scope <= 105:
            break
            
    if best_result is None:
        # Fallback: Simple greedy to ensure we return *something* valid
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last_val = row[-1]
                found = False
                for x in range(last_val + 1, 300):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    # Should not happen with large bound
                    break
            rows.append(row)
        return rows
        
    return best_result
