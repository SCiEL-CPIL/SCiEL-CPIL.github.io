import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0
    
    # Lower bound estimation
    # Total differences needed: n_rows * k*(k+1)/2
    # Available differences in [1, M]: M
    # So M >= n_rows * k*(k+1)/2
    # For n_rows=7, k=6: 7 * 21 = 147? No, k*(k+1)/2 is number of diffs per row.
    # Wait, row has k+1 elements. Number of pairs is (k+1)*k/2.
    # For k=6, pairs = 21.
    # Total diffs = 7 * 21 = 147.
    # So M must be at least 147?
    # The task says "largest entry as small as possible... best known 112".
    # If M=112, available diffs are 1..112. Total 112.
    # But we need 147 distinct diffs?
    # Ah, the differences are $a_j - a_i$.
    # If the max entry is 112, the max difference is 112.
    # The set of differences is a subset of $\{1, \dots, 112\}$.
    # Size of this set is 112.
    # But we need 147 distinct differences?
    # This implies my counting bound logic or the task description implies something else.
    # "simple counting bound is 105".
    # If the bound is 105, and best is 112, it means the number of available slots (105) is *less* than the number of required distinct differences?
    # No, if bound is 105, it means you *need* at least 105.
    # If we need 147 distinct differences, and the max difference is M, we have M possible values.
    # So M must be >= 147.
    # Why is the bound 105?
    # Maybe the differences don't have to be distinct *across* all rows?
    # "all differences between two entries of the same row are distinct over all rows"
    # This means the set of ALL differences from ALL rows must be disjoint.
    # So yes, total distinct diffs = sum over rows of (k+1)k/2.
    # For k=6, row size 7. Pairs = 21.
    # 7 rows * 21 = 147 distinct differences.
    # If the largest entry is $M$, the largest difference is $M$.
    # So we need at least 147 distinct integers in $[1, M]$.
    # So $M \ge 147$.
    # But the task says "best known set has largest entry 112".
    # And "simple counting bound is 105".
    # This is a contradiction in my understanding of the problem statement or the task parameters.
    # Let's re-read carefully: "Choose 7 rows of 6 increasing integers starting at 0".
    # "6 increasing integers" usually means 6 numbers.
    # If a row has 6 integers, the number of pairs is 6*5/2 = 15.
    # 7 rows * 15 = 105 distinct differences.
    # So M must be at least 105.
    # This matches the "simple counting bound is 105".
    # So k in the function `construct(n_rows, k)` corresponds to the number of integers in the row?
    # "Each row is a list of k + 1 integers".
    # If the task says "7 rows of 6 increasing integers", then the row length is 6.
    # So `k+1 = 6` => `k=5`.
    # BUT the task contract says: "The evaluator passes this task's n_rows and k."
    # And "Difference triangle set (7,5)".
    # Usually (n, k) denotes n rows of k+1 elements? Or n rows of k elements?
    # In difference triangle sets, $DTS(n, k)$ often means $n$ rows, each of size $k+1$?
    # Or size $k$?
    # If (7,5) means 7 rows of size 6 (since 5 is the index?), let's check the counting.
    # If size is 6, pairs = 15. Total = 105. Bound 105. Matches.
    # So the row length is 6.
    # The function signature is `construct(n_rows, k)`.
    # And "Each row is a list of k + 1 integers".
    # If the evaluator passes `k=5`, then row length is 6.
    # If the evaluator passes `k=6`, row length is 7.
    # The task title is "Difference triangle set (7,5)".
    # It is highly likely `n_rows=7` and `k=5` are passed, resulting in rows of length 6.
    # OR `n_rows=7` and `k=6` are passed, resulting in rows of length 7?
    # If rows of length 7, pairs = 21, total 147. Bound 147.
    # But the text says "best known ... 112". 112 is close to 105, not 147.
    # So the problem is almost certainly **7 rows of 6 integers**.
    # This implies the input `k` will be 5.
    # I must ensure my code handles `k` correctly as the parameter for `k+1` length.

    # Strategy:
    # 1. Determine row_len = k + 1.
    # 2. Calculate lower_bound = n_rows * (row_len * (row_len - 1) // 2).
    # 3. Iterate M from lower_bound upwards.
    # 4. For each M, run backtracking to find a valid configuration.
    # 5. Return the first valid configuration found (which will have minimal M).

    row_len = k + 1
    diffs_per_row = row_len * (row_len - 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound on scope M
    # Max difference is M. We need total_diffs_needed distinct values in 1..M.
    # So M >= total_diffs_needed.
    min_scope = total_diffs_needed
    
    # We can also use a tighter bound?
    # For now, start from min_scope.
    
    # Backtracking function
    def solve(max_scope):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row):
            # Time check
            if time.time() - start_time > time_limit:
                return False
            
            # If all rows are complete
            if row_idx == n_rows:
                return True
            
            # If current row is complete, move to next
            if len(current_row) == row_len:
                rows.append(current_row)
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            # Determine candidates for next element
            last_val = current_row[-1]
            # The next element x must satisfy:
            # 1. x > last_val
            # 2. x <= max_scope
            # 3. The differences x - a for a in current_row are not in used_diffs
            # 4. Forward check: We need to fit (row_len - len(current_row) - 1) more elements after x.
            #    The minimum span for these is (row_len - len(current_row) - 1).
            #    So x + (row_len - len(current_row) - 1) <= max_scope.
            #    => x <= max_scope - (row_len - len(current_row) - 1)
            
            remaining_slots = row_len - len(current_row) - 1
            max_x = max_scope - remaining_slots
            
            candidates = []
            
            # Iterate x from last_val + 1 to max_x
            # Heuristic: Try smaller x first to keep the row compact?
            # Or try x that introduces "large" differences?
            # Let's try sorting by the minimum difference introduced.
            # If we introduce a very small difference, it might block future small diffs.
            # But small diffs are needed.
            # Actually, for DTS, it's often beneficial to use small differences early?
            # Let's just try x in increasing order. This is a standard approach for minimization.
            
            for x in range(last_val + 1, max_x + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Additional Pruning:
                # Check if the new diffs allow completion of the *current* row?
                # Simple check: The remaining elements must be > x.
                # Their diffs with x and previous elements must be valid.
                # This is hard to check without recursion.
                
                candidates.append(x)
                
                # Limit candidates to avoid explosion?
                # If we have too many, we might want to prune.
                # But with M ~ 112 and row len 6, the branching factor is manageable.
                # However, to be safe, we can limit to the first N "good" candidates?
                # Since we are iterating in increasing order, the first valid ones are the "smallest".
                # Let's just take the first few if the list gets too long?
                # No, let's keep them all but sort if we use a heuristic.
                # For now, increasing order is a strong heuristic for "minimal scope".
                
            # If we have many candidates, we can prioritize those that use "less scarce" diffs?
            # Scarcity of d is proportional to 1/d?
            # If we use d=1, we lose 1. If we use d=100, we lose 100.
            # Small d are more "valuable" for tight packing?
            # Actually, if we use a small d now, we can't use it later.
            # If we use a large d now, we save small d for later.
            # Later rows might need to be tighter?
            # Let's stick to increasing x. It's simple and effective for finding *a* solution with small M.
            
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                # Update state
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs
                
                if backtrack(row_idx, current_row + [x]):
                    return True
                
                # Restore state
                used_diffs = old_used
            
            return False
        
        return backtrack(0, [0])

    # Iterative Deepening
    # Start from min_scope and go up.
    # To avoid getting stuck in a very hard instance at the lower bound,
    # we can add a time limit per M?
    # Or just run until time limit.
    
    # Optimization: If we fail at M, we don't know if it's impossible or just hard.
    # But if we find a solution at M, it's optimal (since we started from lower bound).
    
    # We should also consider that the lower bound might be too tight due to structural constraints.
    # We'll just iterate M.
    
    # To speed up, we can start M a bit higher if min_scope is very low?
    # No, the gap is usually small.
    
    # One issue: If solve(M) takes 100s and fails, we have 15s left.
    # We should limit the time per M?
    # Let's set a soft limit per M, e.g., 5s?
    # If we exceed it, we assume M is too small or hard, and try M+1?
    # But if M is the optimal one and it's just hard, we might skip it.
    # However, for DTS(7,5), the optimal is 112. Lower bound 105.
    # The search space for 105-111 is likely intractable or empty.
    # 112 is the target.
    
    # Let's try M from min_scope to min_scope + 20.
    # If not found, increase range?
    
    # Better: Use a global time limit.
    # If we are at M and time runs out, we return the best found so far?
    # But we need to return a valid solution.
    # If we haven't found one, we must find one.
    # So we should ensure we eventually find a valid one.
    # We can start with a large M (e.g., 200) to get a valid solution quickly,
    # then try to improve it?
    # The current program did this.
    # But the directive says "minimizes the largest entry".
    # And "Score: ... minimized".
    # So we need the *best* valid solution within time.
    
    # Strategy:
    # 1. Find *any* valid solution with a generous bound (e.g., 200) to have a fallback.
    # 2. Try to improve it by searching for solutions with smaller M, starting from lower bound.
    # 3. Keep the best found.
    
    best_result = None
    best_scope = float('inf')
    
    # Step 1: Find a valid solution quickly
    # Use a greedy or simple backtracking with high M.
    # Let's just use the backtracking with M=200.
    # But backtracking might be slow.
    # A greedy construction is fast.
    
    def greedy_construct(max_val):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                found = False
                # Try to find smallest x > row[-1] such that diffs are new
                for x in range(row[-1] + 1, max_val + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    return None # Failed
            rows.append(row)
        return rows
    
    # Try greedy with a reasonable max_val (e.g., 150)
    # If greedy fails, increase max_val.
    current_max = 150
    while time.time() - start_time < time_limit - 5: # Leave 5s for improvement
        res = greedy_construct(current_max)
        if res:
            scope = max(r[-1] for r in res)
            if scope < best_scope:
                best_scope = scope
                best_result = res
            # Try to improve with backtracking starting from scope - 1
            break
        current_max += 10
    
    if best_result is None:
        # Greedy failed, try backtracking with high M
        if solve(200):
             # solve doesn't return the rows, it modifies 'rows' in closure?
             # My solve function above uses 'rows' in closure.
             # I need to refactor solve to return the result.
             pass # Handled below

    # Refactor solve to return the solution
    def solve_for_scope(max_scope):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row):
            if time.time() - start_time > time_limit:
                return None
            
            if row_idx == n_rows:
                return [r[:] for r in rows]
            
            if len(current_row) == row_len:
                rows.append(current_row)
                res = backtrack(row_idx + 1, [0])
                if res:
                    return res
                rows.pop()
                return None
            
            last_val = current_row[-1]
            remaining_slots = row_len - len(current_row) - 1
            max_x = max_scope - remaining_slots
            
            # Heuristic: Try candidates in increasing order of x
            # This tends to find compact rows.
            
            for x in range(last_val + 1, max_x + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs
                
                res = backtrack(row_idx, current_row + [x])
                if res:
                    return res
                
                used_diffs = old_used
            
            return None
        
        return backtrack(0, [0])

    # If greedy didn't find anything, or we want to improve
    # We have best_result from greedy (if found).
    # Now try to find a better one (smaller scope) using backtracking.
    # Start from min_scope and go up to best_scope - 1.
    
    # To avoid wasting time on intractable small scopes, we can limit the time per scope.
    # But we have a global time limit.
    
    # If we have a good best_scope, we can try to verify if smaller ones exist.
    # Let's iterate M from min_scope to best_scope - 1.
    
    # Optimization: If best_scope is already close to min_scope, stop.
    
    for M in range(min_scope, best_scope):
        if time.time() - start_time > time_limit - 2:
            break
            
        # Try to find solution with scope M
        # To prevent getting stuck, we can add a node limit or time limit per M.
        # Let's just rely on the global time limit in backtrack.
        
        res = solve_for_scope(M)
        if res:
            best_result = res
            best_scope = M
            # Since we are iterating upwards, and we found a solution at M,
            # and M > min_scope, it's the best so far.
            # But wait, we started from min_scope. So the first M we find is the optimal?
            # Yes, if we check every M from min_scope up.
            # So we can break?
            # Only if we are confident that M is the *first* one that works.
            # If we skip M because it was too hard (timeout), we might miss it.
            # But if we don't timeout, and find it, it is optimal.
            # So we can break.
            break

    if best_result is None:
        # Fallback: return a valid solution even if not optimal
        # Try a very large scope
        res = solve_for_scope(300)
        if res:
            return res
        else:
            # Should not happen
            return [[0]] * n_rows

    return best_result
