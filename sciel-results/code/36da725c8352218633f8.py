import time
import sys
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
    # We need to construct n_rows rows, each of length k+1.
    # Each row starts with 0.
    # All differences a[j] - a[i] (i < j) across all rows must be distinct.
    
    # Lower bound on M (max entry):
    # Total distinct differences needed: n_rows * k * (k+1) / 2
    # Differences are in {1, ..., M}. So M >= total_diffs.
    # Also, the row itself has k+1 elements in [0, M], so k+1 <= M+1 => M >= k.
    total_diffs_needed = n_rows * k * (k + 1) // 2
    # Simple counting bound: M >= total_diffs_needed
    # But actually, the differences are just a subset of {1, ..., M}.
    # So M must be at least total_diffs_needed.
    M_lower_bound = max(k, total_diffs_needed)
    
    # The best known for (4,7) is 134.
    # Counting bound: 4 * 7 * 8 / 2 = 112.
    # So we search for M in [M_lower_bound, M_lower_bound + 50] or similar.
    # To be safe and fast, we can try to find the minimal M.
    
    # However, the task asks to minimize the largest entry.
    # We will perform a binary search or linear search on M (the scope)
    # to find the smallest M for which a valid configuration exists.
    # Given the time limit, we should be careful with the upper bound.
    
    # Let's estimate a reasonable upper bound.
    # Greedy usually gives something.
    # For (4,7), 134 is known.
    # Let's try M from M_lower_bound to M_lower_bound + 100.
    
    # Optimization: 
    # We will use backtracking to build rows one by one.
    # State: 
    #   rows: list of completed rows
    #   used_diffs: set of all differences used so far
    #   current_row: list of elements in the row being built
    #   current_max: max element in current_row
    
    # Heuristic:
    #   When adding the next element to current_row, try the smallest possible value 
    #   that doesn't conflict with used_diffs and maintains strict increase.
    #   This "greedy smallest" approach tends to keep the numbers small.
    
    # To avoid getting stuck, we can add some randomization or restarts.
    # But first, let's try a deterministic greedy-with-backtracking.
    
    # Since the search space is large, pure backtracking might be slow.
    # We will implement a limited backtracking search.
    
    # Let's define the search for a specific M.
    # If a valid set is found with max entry <= M, we can try smaller M.
    
    # Actually, the "smallest possible value" heuristic during construction
    # naturally tries to minimize the max entry.
    # If we just run the backtracking with "try smallest valid value first",
    # the first solution found is likely to have a small max entry.
    
    # Let's refine the strategy:
    # 1. Start with M = M_lower_bound.
    # 2. Try to construct the set with max entry <= M.
    # 3. If successful, return it.
    # 4. If not, increment M and retry.
    # 5. Stop if time runs out or M exceeds a reasonable cap.
    
    # However, "max entry <= M" is a constraint.
    # The backtracking must respect this.
    
    # Let's implement the backtracking solver for a fixed M.
    
    def solve_with_bound(M):
        """
        Tries to find a valid configuration where all entries are <= M.
        Returns the list of rows if found, else None.
        """
        used_diffs = set()
        rows = []
        
        # We build rows one by one.
        # Symmetry breaking: The first row can be assumed to be the "greedy" one?
        # Not necessarily. But we can fix the first element of each row to 0.
        
        # To improve speed, we can order the rows.
        # Since all rows start with 0, and differences must be distinct,
        # there is no obvious ordering between rows.
        
        # Let's use recursion with backtracking.
        
        def backtrack(row_idx, current_row):
            # Check time
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                # All rows constructed
                return True
            
            # If we are starting a new row, current_row should be [0]
            # If we are continuing a row, current_row has elements.
            
            # We need to complete the current_row to length k+1.
            # Then move to next row.
            
            # Optimization: If current_row is full, push it and start next.
            if len(current_row) == k + 1:
                # Commit this row
                rows.append(current_row)
                # Update used_diffs
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        used_diffs.add(current_row[j] - current_row[i])
                
                # Start next row
                if backtrack(row_idx + 1, [0]):
                    return True
                
                # Backtrack: undo
                rows.pop()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        used_diffs.remove(current_row[j] - current_row[i])
                
                return False
            
            # Else, we need to add an element to current_row.
            # The next element must be > current_row[-1] and <= M.
            
            last_val = current_row[-1]
            
            # Try values from last_val + 1 to M
            # Heuristic: Try smaller values first to keep scope small.
            # But if we always try smallest, we might get stuck deep in the search.
            # We can add a small randomization or just rely on the bound M.
            
            # To speed up, we can prune if the number of remaining slots 
            # cannot be filled with distinct differences?
            # That's hard to compute quickly.
            
            # Let's just iterate.
            # Optimization: If M is small, the range is small.
            
            for v in range(last_val + 1, M + 1):
                # Check if v creates any conflict with used_diffs
                # Differences with existing elements in current_row
                conflict = False
                new_diffs = []
                for a in current_row:
                    d = v - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if conflict:
                    continue
                
                # Also check if new_diffs have duplicates within themselves?
                # No, since current_row is strictly increasing, v - a is strictly decreasing as a increases.
                # So v - current_row[i] are all distinct.
                
                # Tentatively add
                current_row.append(v)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if backtrack(row_idx, current_row):
                    return True
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False
        
        # Start with first row [0]
        return backtrack(0, [0])
    
    # Now, search for the minimal M.
    # Start from M_lower_bound.
    # If we find a solution, return it.
    # If not, increment M.
    
    # To handle potential slowness of backtracking for small M (if it fails),
    # we might want to limit the search depth or time per M.
    # But the global deadline is the main guard.
    
    # Let's try M from M_lower_bound upwards.
    # For (4,7), M_lower_bound is 112.
    # We expect a solution around 134.
    # So we might try 112, 113, ..., 134.
    # Each attempt might be fast if it fails early, or slow if it explores deeply.
    
    # To be safe, we can interleave:
    # Try M = M_lower_bound for 10 seconds.
    # If fail, try M = M_lower_bound + 1 for 10 seconds.
    # ...
    
    # However, a single long backtracking might find it.
    # Let's just loop M and let the global deadline stop us.
    
    # One issue: Backtracking might get stuck in a local minimum for a specific M
    # and waste time.
    # Adding a "kick" or random restart within the M-search could help.
    
    # Let's implement a simple loop.
    
    # Estimate upper bound to avoid infinite loop if no solution found.
    # Best known is 134. Let's go up to M_lower_bound + 100.
    # Or just until deadline.
    
    best_solution = None
    
    # We can also try a greedy construction first to get a valid (but maybe larger) solution.
    # This ensures we have *something* to return if the search fails.
    
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    valid = True
                    for a in row:
                        if x - a in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        break
                    x += 1
            rows.append(row)
        return rows
    
    # Get a baseline solution
    baseline = greedy_construct()
    baseline_scope = max(max(r) for r in baseline)
    
    # Now try to improve with backtracking.
    # Start M from M_lower_bound.
    # If we find a solution with scope < baseline_scope, update baseline.
    
    # To avoid spending all time on M values that are too small and impossible,
    # we can add a heuristic: if M < M_lower_bound, skip.
    
    # Let's run the backtracking search.
    # We will iterate M.
    
    # To make it more robust, we can randomize the order of trying values in the backtracking?
    # No, "smallest first" is best for minimizing scope.
    
    # Let's run it.
    
    # Optimization: 
    # If n_rows * k * (k+1) / 2 > M, it's impossible.
    # This is already handled by M_lower_bound.
    
    # Let's try M values.
    # We'll keep track of the best solution found.
    
    # Since backtracking is deterministic, it will either find a solution or not for a given M.
    # We don't need random restarts for the *existence* check, but we might need them
    # if the search space is too large for a specific M and it times out.
    
    # Let's add a timeout per M attempt?
    # No, just use the global deadline.
    
    # Start M search
    current_M = M_lower_bound
    
    # We can jump M if we are confident.
    # But let's just increment.
    
    # To speed up, we can pre-compute that if M is too small, it fails fast.
    
    while time.time() < deadline - 2:
        if current_M > baseline_scope:
            # We found a better solution than greedy? 
            # If we found a solution with M < baseline_scope, we would have updated baseline_scope.
            # If we are here, it means we tried M up to baseline_scope and didn't find anything better?
            # Or we found something and updated baseline_scope, so current_M > baseline_scope means we are done?
            # Wait, if we find a solution with M, we update baseline_scope to M.
            # Then we continue to M+1? No, we want the *smallest* M.
            # So if we find a solution with M, we should return it immediately?
            # Yes, because we are iterating M upwards. The first M that works is the minimum.
            
            # So if we found a solution in the previous iteration, we should have returned.
            # The only reason we are here is if we didn't find a solution yet.
            # If current_M > baseline_scope, it means we tried all M up to baseline_scope and failed.
            # This implies the greedy solution is the best we can do with this method?
            # Or the method failed to find a solution for M < baseline_scope.
            # In that case, we should break and return the baseline.
            break
        
        # Try to solve with current_M
        # We need to pass the deadline to the solver?
        # The solver uses the global deadline.
        
        # We need to clear the global state (used_diffs, rows) for each M attempt.
        # The solve_with_bound function uses local variables, so it's clean.
        
        # However, solve_with_bound as defined above uses `rows` and `used_diffs` from the closure?
        # No, I defined them inside solve_with_bound.
        # But `backtrack` is nested inside `solve_with_bound`.
        # So it's clean.
        
        # Let's run it.
        # Note: solve_with_bound returns True/False in my draft, but I want the rows.
        # I need to modify it to return the rows.
        
        # Let's rewrite solve_with_bound to return the solution.
        
        def solve_with_bound_return(M):
            used_diffs = set()
            rows = []
            
            def backtrack(row_idx, current_row):
                if time.time() > deadline:
                    return None
                
                if row_idx == n_rows:
                    return rows[:]
                
                if len(current_row) == k + 1:
                    rows.append(current_row)
                    for i in range(len(current_row)):
                        for j in range(i + 1, len(current_row)):
                            used_diffs.add(current_row[j] - current_row[i])
                    
                    res = backtrack(row_idx + 1, [0])
                    if res is not None:
                        return res
                    
                    rows.pop()
                    for i in range(len(current_row)):
                        for j in range(i + 1, len(current_row)):
                            used_diffs.remove(current_row[j] - current_row[i])
                    return None
                
                last_val = current_row[-1]
                
                for v in range(last_val + 1, M + 1):
                    conflict = False
                    new_diffs = []
                    for a in current_row:
                        d = v - a
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.append(d)
                    
                    if conflict:
                        continue
                    
                    current_row.append(v)
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    res = backtrack(row_idx, current_row)
                    if res is not None:
                        return res
                    
                    current_row.pop()
                    for d in new_diffs:
                        used_diffs.remove(d)
                
                return None
            
            return backtrack(0, [0])
        
        solution = solve_with_bound_return(current_M)
        
        if solution is not None:
            # We found a valid solution with max entry <= current_M.
            # Since we iterate M upwards, this is the minimal M.
            return solution
        
        current_M += 1
        
        # If we are taking too long on one M, we might want to skip some?
        # No, just let the deadline handle it.
    
    # If we didn't find a better solution, return the greedy one.
    return baseline

