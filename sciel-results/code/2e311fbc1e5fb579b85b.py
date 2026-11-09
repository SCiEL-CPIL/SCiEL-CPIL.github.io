import time
import sys
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Total number of differences per row: k(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # 1. Greedy Initial Solution
    # This ensures we have a valid starting point.
    def greedy_initial():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows

    initial_rows = greedy_initial()
    best_rows = [r[:] for r in initial_rows]
    best_scope = max(max(r) for r in best_rows)
    
    # CSP Backtracking Solver
    # Variables: (row_idx, diff_idx) -> value
    # But it's easier to think in terms of constructing rows one by one, 
    # or even better, assigning differences to specific "slots" in the rows.
    # However, the structure of a row is rigid: if we know the differences, 
    # can we reconstruct the row?
    # A row is defined by k elements x_1...x_k. 
    # Differences are x_j - x_i.
    # If we fix x_1...x_k, differences are fixed.
    # If we fix differences, x's are determined up to a shift (but x_0=0 fixes shift).
    # Actually, if we know all pairwise differences, the row is uniquely determined 
    # (relative to 0).
    
    # Strategy:
    # We will perform a backtracking search over the rows.
    # We maintain a global set of used differences.
    # For each row, we need to find a set of k integers (plus 0) such that 
    # all pairwise differences are in the "available" pool (not used yet) and 
    # are distinct within the row (implied by strictly increasing).
    
    # Heuristic:
    # 1. Try to construct rows one by one.
    # 2. Within a row, use backtracking to place elements.
    # 3. To improve scope, we try to keep numbers small.
    # 4. We can use "Randomized Restart" with different random seeds/orderings 
    #    to explore different regions of the solution space.
    
    # To beat the greedy solution, we need to find a configuration where 
    # the maximum element is smaller.
    
    # Let's implement a solver that tries to find a solution with scope <= limit.
    # We can binary search on the limit, or just try to minimize it via restarts.
    
    def solve_with_limit(limit):
        """
        Attempts to find a valid set of rows where max element <= limit.
        Returns list of rows if found, None otherwise.
        """
        used_diffs = set()
        rows = []
        
        # Order of rows to solve: We can try to solve the "hardest" rows first?
        # Or just solve them in order.
        # Since all rows have the same structure (k+1 elements), symmetry is high.
        # But the "used_diffs" set grows, making later rows harder.
        
        for r_idx in range(n_rows):
            # Backtracking for this row
            # State: current row list, current max index
            # We want to pick x_1, x_2, ..., x_k such that:
            # 0 < x_1 < ... < x_k <= limit
            # All diffs x_j - x_i are not in used_diffs and distinct.
            
            row = [0]
            
            def backtrack_row(current_row, next_val_min):
                if len(current_row) == k + 1:
                    # Check if all diffs are valid (they are by construction)
                    # Update global used_diffs
                    new_diffs = set()
                    for i in range(len(current_row)):
                        for j in range(i+1, len(current_row)):
                            d = current_row[j] - current_row[i]
                            if d in used_diffs:
                                return False
                            new_diffs.add(d)
                    # Commit
                    used_diffs.update(new_diffs)
                    rows.append(current_row)
                    return True
                
                # MRV Heuristic: 
                # The "variable" is the next element x.
                # Constraints: x > last element, x <= limit.
                # Also, x - last_element must not be in used_diffs.
                # And all x - prev_elements must not be in used_diffs.
                
                # To minimize scope, we should try smaller x first.
                # But sometimes small x leads to dead ends.
                # We can randomize the order slightly to escape local minima.
                
                start_val = next_val_min
                end_val = limit
                
                # Generate candidate values
                # Optimization: We can skip values that definitely cause collision with used_diffs
                # d = x - a. If d in used_diffs, x is invalid.
                
                # Randomization: With small probability, try a larger value first?
                # Or shuffle a window of candidates?
                
                candidates = list(range(start_val, end_val + 1))
                
                # Heuristic: Prioritize candidates that don't create "small" differences 
                # if small differences are already used? 
                # Actually, just trying in order is good for minimizing max.
                # But to avoid getting stuck, we can do a "lookahead" or random restart.
                
                # To speed up, we can check validity quickly.
                for x in candidates:
                    # Quick check: x - last element
                    last = current_row[-1]
                    d_last = x - last
                    if d_last in used_diffs:
                        continue
                    
                    # Check against all previous
                    valid = True
                    for a in current_row[:-1]:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Recurse
                    current_row.append(x)
                    if backtrack_row(current_row, x + 1):
                        return True
                    current_row.pop()
                
                return False
            
            if not backtrack_row(row, 1):
                return None
        
        return rows

    # Strategy:
    # 1. Start with the greedy solution's scope.
    # 2. Try to find a solution with scope = best_scope - 1.
    # 3. If found, update best_scope.
    # 4. Repeat until time runs out or no improvement.
    # 5. Add randomization to the backtracking to explore different solutions 
    #    for the same scope, which might allow further reductions later.
    
    current_limit = best_scope
    
    # To make the search more effective, we can introduce randomization in the candidate order
    # within the backtracking.
    
    def randomized_solve_with_limit(limit, seed):
        """
        Same as solve_with_limit but with randomized candidate order to explore
        different solutions within the same scope.
        """
        random.seed(seed)
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            
            def backtrack_row(current_row, next_val_min):
                if len(current_row) == k + 1:
                    new_diffs = set()
                    for i in range(len(current_row)):
                        for j in range(i+1, len(current_row)):
                            d = current_row[j] - current_row[i]
                            if d in used_diffs:
                                return False
                            new_diffs.add(d)
                    used_diffs.update(new_diffs)
                    rows.append(current_row)
                    return True
                
                last = current_row[-1]
                start_val = next_val_min
                end_val = limit
                
                # Generate candidates
                # We want to try small values first to keep scope low, 
                # but randomize slightly to avoid deterministic dead-ends.
                
                # Heuristic: 
                # 1. Check if x - last is in used_diffs.
                # 2. Check if x - any_prev is in used_diffs.
                
                # To speed up, we can generate a list of valid candidates first?
                # That might be expensive. Let's just iterate.
                
                # Randomized order:
                # Take a window of size W from the start.
                # Shuffle it.
                # Then append the rest in order.
                # This allows local exploration while maintaining bias towards small values.
                
                window_size = 20
                candidates = []
                for x in range(start_val, end_val + 1):
                    # Quick filter
                    d_last = x - last
                    if d_last in used_diffs:
                        continue
                    valid = True
                    for a in current_row[:-1]:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                
                if not candidates:
                    return False
                
                # Shuffle the first part
                if len(candidates) > window_size:
                    head = candidates[:window_size]
                    random.shuffle(head)
                    candidates = head + candidates[window_size:]
                else:
                    random.shuffle(candidates)
                
                for x in candidates:
                    current_row.append(x)
                    if backtrack_row(current_row, x + 1):
                        return True
                    current_row.pop()
                
                return False
            
            if not backtrack_row(row, 1):
                return None
        
        return rows

    # Main Loop
    # We try to reduce the limit.
    # If we can't reduce it, we try to find alternative solutions at the current limit
    # that might be "better" in terms of structure, though score only depends on max.
    # Actually, if we find a solution with limit L, we can immediately try L-1.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Try to improve the limit
        # We can try decreasing the limit by 1.
        # If that fails, we can try randomizing at the current limit to see if 
        # a different solution structure allows a reduction? 
        # No, if solve_with_limit(L-1) fails, it's because no solution exists with max <= L-1.
        # So we must find a solution with max <= L-1.
        
        # However, the backtracking might get stuck in a local minimum due to 
        # the order of rows or the specific path taken.
        # So, randomized restarts at the target limit are useful.
        
        target_limit = best_scope - 1
        
        if target_limit < k: # Theoretical minimum is k (0..k) but diffs must be distinct
            # 0..k has diffs 1..k. Next row needs diffs not in 1..k.
            # Min max is definitely > k.
            pass
            
        # Try a few random seeds for the target limit
        found = False
        for seed in range(5): # Try 5 randomizations per limit step
            if time.time() > deadline:
                break
                
            result = randomized_solve_with_limit(target_limit, seed + attempts * 100)
            if result:
                best_scope = target_limit
                best_rows = result
                found = True
                break
        
        if not found:
            # If we can't find a solution with best_scope - 1, 
            # maybe we can find a different solution with best_scope 
            # that is "closer" to being reducible? 
            # Or just accept that we might be at the optimum for this search strategy.
            # To avoid wasting time, we can try a few randomizations at current best_scope
            # just to ensure we have a valid solution (we already do) 
            # and maybe find one that allows a future reduction if we change row order?
            # But the problem is symmetric with respect to row order.
            
            # Let's try to reduce the limit by 2? No, that's too aggressive.
            # Let's just stop trying to reduce if we've failed a few times?
            # Or continue trying with different seeds?
            
            # If we are stuck, we might want to try a "relaxation" approach?
            # But for now, let's just keep trying the -1 step with new seeds.
            # If it fails, we haven't improved.
            # To prevent infinite loop of failures, we can add a small delay or 
            # just let the time limit handle it.
            pass

    return best_rows
# EVOLVE-BLOCK-END
