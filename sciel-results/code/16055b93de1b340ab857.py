import random
import time
import math

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    best_result = None
    best_scope = float('inf')
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer for the evaluator
    
    # Parameters for the search
    # We try to find a solution with scope <= counting_bound + margin
    # Counting bound for (7,5) is 105.
    # Let's aim for 105 to 115.
    
    max_restarts = 2000
    
    # Heuristic weight for differences
    # We penalize small differences more heavily.
    # score = sum(weight[d] for d in new_diffs)
    # Lower score is better.
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        used_diffs = set()
        rows = []
        
        # Dynamic upper bound for the current search
        # Start with a bound slightly above the known best or counting bound
        # If we find a better one, we tighten the bound.
        current_bound = 112 if restart == 0 else min(best_scope - 1, 112)
        
        success = True
        
        # Backtracking search for each row
        # We build rows sequentially.
        for row_idx in range(n_rows):
            row = [0]
            # Use a stack or recursive function for backtracking within the row
            # Since k is small (5), depth is small.
            
            # We'll use a simple recursive helper or iterative backtracking
            # Let's use a stack of states: (row, index_in_row)
            # State: current row, next position to fill
            
            # Actually, since k is small, we can just use a nested loop with backtracking
            # But standard greedy with restarts is often faster if the heuristic is good.
            # The directive asks for backtracking.
            
            # Let's implement a depth-first search for the single row, 
            # given the current used_diffs.
            
            def dfs_row(current_row, pos, remaining_diffs_allowed):
                # pos is the index of the next element to add (1 to k)
                if pos == k + 1:
                    return current_row[:]
                
                last_val = current_row[-1]
                
                # Determine upper bound for the next value
                # It must be < current_bound
                # Also, it must leave room for the remaining elements?
                # Not strictly, but good to have a heuristic.
                
                candidates = []
                
                # Iterate possible values for the next element
                # Start from last_val + 1
                # Upper bound: current_bound
                # Optimization: We can prune if last_val + (k - pos) * min_gap > current_bound
                # Min gap is 1.
                
                for x in range(last_val + 1, current_bound + 1):
                    new_diffs = {x - a for a in current_row}
                    
                    # Check if any new diff is already used
                    if new_diffs & used_diffs:
                        continue
                    
                    # Calculate heuristic score
                    # We want to minimize the "cost" of used differences.
                    # Cost is higher for smaller differences.
                    # score = sum(1.0 / d for d in new_diffs)
                    # This encourages using large differences.
                    
                    score = 0.0
                    for d in new_diffs:
                        score += 1.0 / d
                    
                    # Add some randomness to break ties and explore
                    # noise = random.random() * 0.1
                    # score += noise
                    
                    candidates.append((score, x, new_diffs))
                
                if not candidates:
                    return None
                
                # Sort by score (ascending)
                candidates.sort(key=lambda item: item[0])
                
                # Try candidates in order of best score
                # To add diversity, we can shuffle the top N candidates
                # or pick randomly from the top few.
                
                # Let's try the best one first, then others if it fails.
                # For efficiency in this restart-based approach, we might just pick the best
                # or a random one from the top 5.
                
                # To satisfy "backtracking", we should try the best, and if it leads to a dead end
                # later in the row, try the next best.
                
                for score, x, new_diffs in candidates:
                    # Recurse
                    next_row = dfs_row(current_row + [x], pos + 1, remaining_diffs_allowed)
                    if next_row is not None:
                        return next_row
                
                return None
            
            # Call DFS for this row
            # Note: used_diffs is shared across rows.
            # The DFS needs to update used_diffs when it succeeds, and rollback if it fails?
            # Since we are building rows sequentially, and if a row fails, the whole restart fails,
            # we don't need complex rollback across rows.
            # But within a row, if we pick x, and then fail at pos+1, we need to try next x.
            # The DFS above does this correctly: it returns None if no path works.
            
            # However, the `used_diffs` set is modified in the main loop, not inside DFS.
            # The DFS only checks `used_diffs` (from previous rows) and local `current_row`.
            # It does NOT add to `used_diffs` until the row is complete and accepted.
            # This is correct because differences within the same row are distinct by construction
            # (strictly increasing, so x - a_i are all distinct for fixed x, and distinct across x? 
            # No, x1 - a_i and x2 - a_j can be equal. 
            # Wait, the condition is "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows".
            # This means differences within a row must be distinct from each other? 
            # "distinct over all rows" usually means the global set of all differences from all rows is a set (no duplicates).
            # Yes, "all ... are distinct" implies global uniqueness.
            # So, differences within a single row must also be distinct from each other?
            # If row is [0, 2, 5], diffs are 2, 5, 3. They are distinct.
            # If row is [0, 2, 4], diffs are 2, 4, 2. Duplicate 2. Invalid.
            # So yes, differences within a row must be distinct.
            # My DFS check `if new_diffs & used_diffs` only checks against PREVIOUS rows.
            # I also need to ensure `new_diffs` are distinct among themselves?
            # `new_diffs` is a set, so it automatically removes duplicates within the set.
            # But `x - a_i` for `a_i` in `current_row`.
            # If `current_row` is `[0, 2]` and `x=4`, `new_diffs = {4, 2}`.
            # If `current_row` is `[0, 2]` and `x=2` (not allowed, strictly increasing).
            # If `current_row` is `[0, 3]` and `x=5`, `new_diffs = {5, 2}`.
            # The set `new_diffs` ensures no internal duplicates in the *new* diffs.
            # But what about overlap between `new_diffs` and diffs *already in current_row*?
            # `current_row` diffs are not in `used_diffs` yet.
            # So I must check `new_diffs` against `current_row`'s existing diffs too.
            
            # Let's fix the DFS to track local diffs.
            
            pass # Re-implementing DFS with local diff tracking

        # The previous DFS sketch was incomplete. Let's write a cleaner one.
        
        success = True
        rows = []
        
        for row_idx in range(n_rows):
            # Backtracking search for this row
            # State: current_row, pos
            # We need to know all diffs in current_row to check conflicts.
            
            def solve_row(pos, current_row, current_diffs):
                if pos == k + 1:
                    return current_row[:]
                
                last_val = current_row[-1]
                
                candidates = []
                
                for x in range(last_val + 1, current_bound + 1):
                    new_diffs = {x - a for a in current_row}
                    
                    # Check conflict with global used_diffs
                    if new_diffs & used_diffs:
                        continue
                    
                    # Check conflict with local current_diffs
                    if new_diffs & current_diffs:
                        continue
                    
                    # Calculate heuristic
                    score = sum(1.0 / d for d in new_diffs)
                    
                    candidates.append((score, x, new_diffs))
                
                if not candidates:
                    return None
                
                candidates.sort(key=lambda item: item[0])
                
                # Randomization: pick from top 3 with probability
                # Or just iterate?
                # To be a proper backtracking, we should try them in order.
                # But to avoid getting stuck in a local minimum of the heuristic,
                # we can shuffle the candidates slightly.
                
                # Let's try the best one. If it fails, the recursion will return None,
                # and we will try the next one in the loop.
                
                for score, x, new_diffs in candidates:
                    next_row = solve_row(pos + 1, current_row + [x], current_diffs | new_diffs)
                    if next_row is not None:
                        return next_row
                
                return None
            
            new_row = solve_row(1, [0], set())
            
            if new_row is None:
                success = False
                break
            
            # Update global used_diffs
            row_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    row_diffs.add(new_row[j] - new_row[i])
            
            used_diffs |= row_diffs
            rows.append(new_row)
            
            # Update current_bound if this row's max is smaller?
            # No, current_bound is a global constraint for the whole set.
            # We can tighten it if we have found a valid partial solution?
            # Not really, because subsequent rows must also fit.
            # But if max(rows) < current_bound, we can try to reduce current_bound?
            # It's safer to keep current_bound fixed for the restart.
            
        if success:
            current_scope = max(r[-1] for r in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_result = [r[:] for r in rows]
                if best_scope <= 105:
                    break

    if best_result is None:
        # Fallback
        used = set()
        rows = []
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
        
    return best_result
