import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    # Total number of distinct differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Simple counting bound: M >= total_diffs
    # However, the structure of difference triangles imposes a stronger bound.
    # The "best known" is 134 for (4,7). total_diffs for (4,7) is 4*28=112.
    # So M is likely between 112 and 134.
    
    # We will attempt to construct a solution with M starting from the lower bound
    # and increasing until we find a valid configuration.
    # Strategy: 
    # 1. Generate a random permutation of difference values from 1..M.
    # 2. Assign these differences to the rows in a way that allows reconstruction.
    # 3. The key insight: A row is defined by its set of differences.
    #    If we pick a set of differences for a row, can we reconstruct the row?
    #    Not uniquely. But we can try to build the row incrementally.
    
    # Alternative Strategy: "Difference-First" Greedy with Backtracking
    # We know that for a valid difference triangle set, the differences in each row
    # must be distinct globally.
    # Let's use a randomized backtracking search on the ROWS, but choosing
    # differences first.
    
    # Actually, a more robust approach for (n_rows, k) is to use a 
    # randomized greedy algorithm that builds rows one by one, but with 
    # "lookahead" and "repair" steps.
    
    # Let's try a different angle: 
    # The problem is equivalent to finding a packing of n_rows copies of the 
    # difference set of a k-gon into [1, M].
    
    # We will use a Simulated Annealing approach on the DIFFERENCES directly.
    # State: An assignment of differences to (row, pair) slots.
    # But pairs are not independent; they must be consistent to form a row.
    
    # Let's stick to a refined "Row-Construction" approach but with a 
    # global constraint check and a "swap" move that is more effective.
    # The previous program used single-element moves. 
    # New Idea: "Row Regeneration" moves.
    # When a row has many conflicts, regenerate the entire row using a 
    # greedy method that avoids already-used differences, constrained by 
    # the current global max M.
    
    def greedy_row(used_diffs, max_val):
        """
        Attempts to build a single row of k+1 integers starting at 0,
        such that all differences are not in used_diffs and max <= max_val.
        Returns the row if successful, None otherwise.
        """
        row = [0]
        # We need to choose k more numbers.
        # To keep max small, we should pick numbers as small as possible?
        # No, picking small numbers creates small differences which are 
        # likely already used.
        # A good heuristic: Pick the smallest available number that doesn't 
        # create a conflict, but also ensure we can finish the row.
        
        # Let's use a randomized greedy with backtracking for the row construction.
        # Since k is small (7), the search space for a single row is manageable.
        
        def backtrack(current_row, used_local):
            if len(current_row) == k + 1:
                return current_row[:]
            
            # Determine the range for the next element
            # It must be > current_row[-1]
            # And current_row[-1] + d <= max_val for some d
            # Also, all differences from new element to previous elements must be valid.
            
            # We want to minimize the max, so we try small values first?
            # But small values create small diffs. 
            # Let's try values in a randomized order within a feasible window.
            
            # Feasible next values:
            # Must be > current_row[-1]
            # Let's try from current_row[-1] + 1 up to max_val.
            # But to be efficient, we can limit the range.
            
            # Actually, if we are building from 0, the gaps determine the structure.
            
            # Let's try a simpler heuristic: 
            # Pick a random starting point for the "gap" sequence?
            
            # For now, let's just try small to large, with some randomization.
            # If we fail, we return None.
            
            # To avoid getting stuck, we can try a few random permutations of 
            # candidate next steps.
            
            candidates = []
            for next_val in range(current_row[-1] + 1, max_val + 1):
                # Check differences
                valid = True
                for prev in current_row:
                    d = next_val - prev
                    if d in used_diffs or d in used_local:
                        valid = False
                        break
                if valid:
                    candidates.append(next_val)
            
            if not candidates:
                return None
            
            # Randomize the order to escape local minima
            random.shuffle(candidates)
            
            # Try the first few candidates
            # Limit depth to avoid explosion, though k is small
            for cand in candidates[:10]: # Try top 10 random candidates
                new_used_local = set(used_local)
                for prev in current_row:
                    new_used_local.add(cand - prev)
                
                res = backtrack(current_row + [cand], new_used_local)
                if res:
                    return res
            
            return None
        
        return backtrack(row, set())

    # Global Search Strategy
    # We will try to find a solution with M starting from the lower bound.
    # We use a randomized restart approach.
    
    best_rows = None
    best_max = float('inf')
    
    # Initial lower bound
    # For (4,7), lower bound is 112.
    # Let's start trying M from 112 upwards.
    
    # To speed up, we can pre-compute that M must be at least total_diffs.
    min_M = total_diffs
    
    # We will run a time-limited search.
    # We'll try increasing M values.
    
    # Since the best known is 134, we expect M to be around there.
    # We'll iterate M from min_M to min_M + 50 (or until time runs out).
    
    # To be safe and fast, we'll use a "Race" strategy:
    # Run multiple threads? No, single process.
    # Run multiple attempts for each M.
    
    # Let's estimate the number of attempts we can afford.
    # If we have 100s, and each attempt takes 1ms, we can do 100k attempts.
    
    # Let's refine the greedy_row to be faster.
    # The backtrack in greedy_row might be slow if max_val is large.
    # But k=7, so depth is 7. Branching factor is limited by conflicts.
    
    # Let's add a "warm start" or "repair" mechanism.
    
    def try_construct_for_M(M, num_attempts):
        used_diffs = set()
        rows = []
        
        for _ in range(num_attempts):
            # Shuffle the order of rows? No, rows are identical in structure.
            # But the assignment of differences matters.
            
            # We'll build rows one by one.
            current_rows = []
            current_used = set()
            success = True
            
            for r in range(n_rows):
                # Try to build a row
                # To increase diversity, we can randomize the greedy choices.
                row = greedy_row(current_used, M)
                if row is None:
                    success = False
                    break
                current_rows.append(row)
                # Update used diffs
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        current_used.add(row[j] - row[i])
            
            if success:
                return current_rows
        
        return None

    # Main Loop
    # We'll try M values from min_M upwards.
    # We'll spend a fraction of the time on each M.
    
    # Since we don't know the exact M, we'll do a binary search or linear scan.
    # Linear scan from min_M is safer.
    
    # To optimize, we can check if a solution exists for M.
    
    # Let's add a heuristic: If we fail for M, we increment M.
    
    # We'll allocate time dynamically.
    # Start with M = min_M.
    # If time is running low, we might jump to a higher M if we know 
    # lower ones are impossible (hard to know).
    
    # Let's just iterate M.
    
    # Optimization: 
    # The greedy_row function is the bottleneck.
    # We can speed it up by pre-computing conflicts? No, dynamic.
    
    # Let's try to improve the initial row construction.
    # A known good heuristic for difference triangles is to use "sparse" rows.
    
    # Let's add a "random restart" inside the M loop.
    
    # We'll use a deadline check inside the M loop.
    
    current_M = min_M
    max_M_to_try = min_M + 50 # Cap the search range
    
    while current_M <= max_M_to_try and time.time() < deadline:
        # Allocate a small time slice for this M
        # E.g., 0.1s per M value initially, then more if it looks promising?
        # For simplicity, we'll just run a fixed number of attempts per M.
        
        # How many attempts?
        # If greedy_row takes ~100us, we can do 1000 attempts in 0.1s.
        
        rows = try_construct_for_M(current_M, 500)
        
        if rows:
            # Verify
            used = set()
            valid = True
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in used:
                            valid = False
                            break
                        used.add(d)
                if not valid: break
            
            if valid:
                best_rows = rows
                best_max = current_M
                # Since we are scanning upwards, the first valid M is the minimum
                # for this specific set of rows. But maybe a different set of rows
                # works for a smaller M? No, we started from min_M.
                # So this is optimal for this search path.
                # However, greedy_row might fail for M even if a solution exists.
                # So we should keep searching for a bit to see if we find a 
                # "better" (lower) M? No, we are going up.
                # So the first M we find is the best we found.
                # We can return immediately?
                # It's possible that a different random seed finds a solution 
                # for the same M, but we already found one.
                # Can we find a solution for a SMALLER M? No, we started at min_M.
                # So if we found a solution for M, it's the minimum possible 
                # for this construction method.
                # But is it the global minimum? 
                # If min_M is the counting bound, and we found a solution at min_M, 
                # it's optimal.
                # If we found it at min_M + 5, maybe min_M + 1 is possible?
                # We should check smaller M values if we jumped? 
                # We are iterating sequentially, so we check min_M, min_M+1, etc.
                # So if we succeed at M, it is the smallest M for which we succeeded.
                # We can return.
                return rows
        
        current_M += 1
        
        # If we are close to deadline, break
        if time.time() > deadline - 5:
            break

    # If we didn't find a solution in the range, return the best effort
    # or a greedy solution with a larger M.
    
    if best_rows:
        return best_rows
    
    # Fallback: Generate a valid solution with a larger M
    # Using the pure greedy from the previous program, which is guaranteed 
    # to find a solution (though M might be large).
    
    def pure_greedy_fallback():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return pure_greedy_fallback()
