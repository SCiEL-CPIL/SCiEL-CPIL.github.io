import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Total number of differences needed
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Lower bound on scope M
    # M(M+1)/2 >= total_diffs => M ~ sqrt(2 * total_diffs)
    import math
    M_lower = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))
    # Add some slack to the lower bound to make finding a solution easier initially
    M_target = M_lower + 5
    
    # We will try to construct rows for increasing M values starting from M_target
    # If we fail, we increase M slightly.
    
    # Bitset for used differences. Size M+1.
    # Since M can be up to ~200-300 for typical small n,k, an integer bitset is fast.
    
    def solve_csp(M):
        """
        Attempts to construct n_rows rows of length k+1 with max entry <= M.
        Uses backtracking with forward checking.
        """
        # Pre-compute potential positions for each row? 
        # A row is a strictly increasing sequence of length k+1 starting at 0, ending <= M.
        
        # To speed up backtracking, we process rows one by one.
        # Within a row, we pick elements one by one.
        # State: 
        #   used_diffs: int bitmask where bit d is set if difference d is used.
        #   current_row: list of elements chosen so far for the current row.
        #   row_idx: which row we are constructing.
        
        # Heuristic: Order rows by "difficulty"? 
        # Actually, since all rows are symmetric, order doesn't matter much for validity, 
        # but it might for search efficiency if we fix a specific structure.
        # Let's just go row by row.
        
        # To make it faster, we can pre-compute valid next steps? 
        # No, dynamic.
        
        # Optimization: If we find a solution, return it.
        # We use recursion with a depth limit or just let it run.
        # To avoid deep recursion stack issues or time limits, we add a step counter.
        
        steps = 0
        max_steps = 1000000  # Hard cap on backtracking steps
        
        # used_diffs bitmask
        # Max difference is M.
        # We need bits 1 to M.
        
        def backtrack(row_idx, current_row, used_diffs):
            nonlocal steps
            
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return None  # All rows constructed, success
            
            # We need to construct the (row_idx + 1)-th row.
            # It must start with 0.
            # Length k+1.
            
            # If current_row is empty, we start a new row.
            # If current_row has elements, we are extending the current row.
            
            if len(current_row) == 0:
                current_row = [0]
            
            if len(current_row) == k + 1:
                # Row complete. Recurse to next row.
                return backtrack(row_idx + 1, [], used_diffs)
            
            # Find the next element in current_row.
            # Last element is current_row[-1].
            # Next element x must satisfy:
            # 1. x > current_row[-1]
            # 2. x <= M
            # 3. For all a in current_row, (x - a) must not be in used_diffs.
            
            last = current_row[-1]
            
            # MRV Heuristic: 
            # We are choosing the next element. 
            # We can try values from last+1 to M.
            # To find a solution quickly, maybe try smaller values first? 
            # Or use a randomization to avoid getting stuck in a local dead end?
            # Let's try a mix: prioritize smaller values (to keep scope low) but with some randomness.
            
            # Actually, for CSP, a good heuristic is to choose the value that leaves the most options for the future?
            # That's expensive. Let's stick to a simple ordering with randomization.
            
            # Generate candidate values
            candidates = list(range(last + 1, M + 1))
            
            # Shuffle slightly to escape deterministic dead ends? 
            # Or just iterate. If we iterate 1..M, we might fill small diffs first, 
            # leaving large diffs for later which might be harder? 
            # Actually, filling small diffs first is usually good because they are more "constrained" 
            # in the sense that they appear more frequently in small ranges?
            # Let's try sorting by value, but with a random jitter.
            
            # For performance, we don't want to create a list of 200 ints every step.
            # We can iterate and check.
            
            # Let's try values in a random order to break symmetry?
            # No, let's try a "Least Constraining Value" heuristic approximation:
            # Prefer values that create differences that are "rare" or "large"?
            # Actually, let's just try all values from last+1 to M, but randomly pick a starting point?
            
            # Let's try a random permutation of candidates to avoid bias.
            random.shuffle(candidates)
            
            # But we want to find a valid solution. Random order might take longer to find the first one.
            # Let's try a hybrid: 50% chance to try in ascending order, 50% random?
            # Or just ascending. If it fails, it fails.
            
            # Let's stick to ascending order for determinism and speed of finding "tight" solutions.
            # But to avoid getting stuck in a specific local minimum, we can randomize the initial state or use multiple restarts.
            
            # Let's iterate candidates.
            for x in candidates:
                # Check constraints
                valid = True
                for a in current_row:
                    d = x - a
                    if used_diffs & (1 << d):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Accept x
                new_used = used_diffs
                for a in current_row:
                    d = x - a
                    new_used |= (1 << d)
                
                # Recurse
                steps += 1
                if steps > max_steps:
                    return None
                
                res = backtrack(row_idx, current_row + [x], new_used)
                if res is not None:
                    return res
            
            return None
        
        # Start backtracking
        # We need to return the actual rows, not just None.
        # Let's modify backtrack to return the list of rows if successful.
        
        # Re-implementing backtrack to return solution
        def backtrack_sol(row_idx, current_row, used_diffs, rows_acc):
            nonlocal steps
            
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return rows_acc[:]
            
            if len(current_row) == 0:
                current_row = [0]
                rows_acc.append(current_row)
            
            if len(current_row) == k + 1:
                # Row complete
                # Move to next row
                rows_acc.pop() # Remove current row from acc temporarily? No, keep it.
                # Wait, rows_acc holds completed rows.
                # Let's pass rows_acc as a list of lists.
                # When we start a new row, we append a new list.
                # When we finish, we move to next row_idx.
                
                # Actually, let's manage rows_acc carefully.
                # rows_acc[row_idx] is the current row being built.
                
                # Let's restart the logic for clarity.
                pass
        
        # Cleaner approach:
        # State: (row_idx, current_row_elements, used_diffs)
        # Return: list of rows if success, else None
        
        def bt(row_idx, cur_row, used):
            nonlocal steps
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return None # Signal success via closure variable? Or return list.
                
            # We need to build row row_idx.
            # cur_row is the list of elements in this row so far.
            # If cur_row is empty, start with [0].
            
            if not cur_row:
                cur_row = [0]
            
            if len(cur_row) == k + 1:
                # This row is done.
                # We need to store it and move to next.
                # But we are in a recursive call. We need to accumulate results.
                # Let's use a global list for the solution.
                pass
        
        # Let's use a global variable for the found solution to avoid passing it around.
        found_solution = None
        
        def bt_inner(row_idx, cur_row, used):
            nonlocal found_solution, steps
            
            if found_solution is not None:
                return
            
            if time.time() > deadline:
                return
            
            if row_idx == n_rows:
                found_solution = rows_found[:]
                return
            
            if not cur_row:
                cur_row = [0]
            
            if len(cur_row) == k + 1:
                # Row complete
                rows_found.append(cur_row[:])
                bt_inner(row_idx + 1, [], used)
                if found_solution is not None:
                    return
                rows_found.pop()
                return
            
            last = cur_row[-1]
            
            # Try candidates
            # Optimization: Pre-compute valid diffs? No.
            
            # Heuristic: Try smaller values first to keep scope tight?
            # But we are constrained by M anyway.
            
            # To improve search, let's try values that don't block many future options.
            # Approximation: Try values that create larger differences? 
            # Or just random order.
            
            # Let's try a random shuffle of the range to break symmetry.
            # But for small M, ascending is fast.
            
            # Let's try ascending.
            for x in range(last + 1, M + 1):
                valid = True
                for a in cur_row:
                    d = x - a
                    if used & (1 << d):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                new_used = used
                for a in cur_row:
                    d = x - a
                    new_used |= (1 << d)
                
                steps += 1
                if steps > max_steps:
                    return
                
                cur_row.append(x)
                bt_inner(row_idx, cur_row, new_used)
                cur_row.pop()
                
                if found_solution is not None:
                    return
        
        rows_found = []
        bt_inner(0, [], 0)
        
        if found_solution:
            return found_solution
        return None

    # Try to solve with increasing M
    # Start from M_target
    # If it takes too long, increase M? 
    # Actually, if M is too small, it will fail quickly (hopefully).
    # If M is large, it might succeed but with a large scope.
    # We want the smallest M.
    
    # Strategy: 
    # 1. Try M_target.
    # 2. If fails or times out, try M_target + 1, etc.
    # But backtracking is expensive. 
    # Better: Run backtracking with a time limit for a specific M.
    # If it doesn't find a solution, try a larger M? 
    # No, if it doesn't find a solution for M, it might be because M is too small OR the search order was bad.
    
    # Given the time constraint, let's try a few M values.
    # M_lower is the theoretical minimum.
    # Let's try M_lower, M_lower+1, ..., up to M_lower + 20.
    
    best_rows = None
    
    for offset in range(0, 30):
        M = M_lower + offset
        # Reset steps
        steps = 0
        max_steps = 200000 # Limit per M attempt
        
        # Re-run solve_csp with this M
        # We need to re-initialize the internal state of solve_csp
        # Let's refactor solve_csp to take M and limits.
        
        # For simplicity, I'll inline the solver logic here to allow fresh state per M.
        
        found_sol = None
        rows_found = []
        
        def bt_inner(row_idx, cur_row, used):
            nonlocal found_sol, steps
            
            if found_sol is not None:
                return
            
            if time.time() > deadline:
                return
            
            if row_idx == n_rows:
                found_sol = rows_found[:]
                return
            
            if not cur_row:
                cur_row = [0]
            
            if len(cur_row) == k + 1:
                rows_found.append(cur_row[:])
                bt_inner(row_idx + 1, [], used)
                if found_sol is not None:
                    return
                rows_found.pop()
                return
            
            last = cur_row[-1]
            
            # Randomize candidate order to avoid systematic failure?
            # Let's try a random shuffle for each call to break symmetry.
            candidates = list(range(last + 1, M + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                valid = True
                for a in cur_row:
                    d = x - a
                    if used & (1 << d):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                new_used = used
                for a in cur_row:
                    d = x - a
                    new_used |= (1 << d)
                
                steps += 1
                if steps > max_steps:
                    return
                
                cur_row.append(x)
                bt_inner(row_idx, cur_row, new_used)
                cur_row.pop()
                
                if found_sol is not None:
                    return
        
        bt_inner(0, [], 0)
        
        if found_sol:
            best_rows = found_sol
            break
    
    # If we didn't find anything with CSP, fall back to a greedy heuristic 
    # or return a valid but potentially suboptimal set.
    # The task requires a valid set. If CSP fails, we need a fallback.
    
    if best_rows is None:
        # Fallback: Greedy construction with large M
        # This ensures we return *something* valid, even if score is low.
        M_fallback = 2000
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
        
        best_rows = rows
    
    return best_rows
