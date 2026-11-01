import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
    # Total differences needed per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # We want to minimize the maximum entry (scope).
    # The maximum difference in a row is at most scope.
    # If scope is M, the available differences are 1..M.
    # We need total_diffs_needed distinct differences.
    # So M must be at least total_diffs_needed.
    # Lower bound: M_min = total_diffs_needed.
    # However, structural constraints might require larger M.
    
    # We will try to construct a valid set for increasing values of M (scope).
    # For a fixed M, we check if it's possible to assign values.
    # But the values must be strictly increasing and start at 0.
    # This is a bit tricky because the "difference" constraint depends on the actual values.
    
    # Alternative Strategy: 
    # Use Backtracking to assign elements one by one.
    # Order of assignment: 
    # We can assign row by row, or element by element.
    # Given the "distinct differences" constraint is global, 
    # assigning all elements of one row first might be better for local consistency,
    # but global conflicts arise.
    
    # Let's try a hybrid: 
    # 1. Determine a target scope M.
    # 2. Use backtracking to fill the rows.
    # 3. If we find a solution, return it.
    # 4. If we fail, increase M and retry.
    # 5. If time runs out, return the best valid solution found (or greedy if none).
    
    # To make backtracking efficient, we need good heuristics.
    # Variable Ordering: 
    # Since rows are independent except for the diff pool, we can treat each row's construction 
    # as a subproblem that consumes from the global diff pool.
    # However, the diff pool is shared. 
    # A better way: Assign all variables (n_rows * k elements, since 0 is fixed) 
    # in an order that maximizes constraint propagation.
    # MRV (Minimum Remaining Values) is hard to compute dynamically for continuous domains.
    # But we can restrict the domain of each element.
    # Element (r, c) has value v. 
    # Constraints:
    # 1. v > v_prev (strictly increasing)
    # 2. v < v_next (strictly increasing)
    # 3. For all j < c, (v - v_j) not in used_diffs
    # 4. For all j > c, (v_j - v) not in used_diffs (if v_j already assigned)
    
    # This is complex. Let's simplify.
    # We will iterate over possible scopes M starting from a lower bound.
    # For each M, we run a backtracking search.
    # To speed up, we use a greedy initialization to find a "close" starting point,
    # then use backtracking to repair/optimize? 
    # No, the prompt asks for a DIFFERENT strategy. 
    # Pure backtracking with forward checking is the requested strategy.
    
    # To avoid exponential explosion, we need strong pruning.
    # Pruning: 
    # If we have assigned some elements, we know the set of used differences.
    # For the next element to be assigned, its value must not create a duplicate difference.
    
    # Let's define the search state:
    # - rows: list of lists, partially filled
    # - used_diffs: set of used differences
    # - current_row, current_col: next element to assign
    # - M: current scope limit
    
    # We will assign elements in row-major order: (0,1), (0,2)...(0,k), (1,1)...
    # This is simple but might not be optimal for MRV.
    # However, implementing dynamic MRV for this specific constraint is hard.
    # Let's stick to row-major but with strong forward checking.
    
    # Forward Checking:
    # Before assigning value v to (r, c), check if it conflicts with:
    # 1. Previously assigned elements in the same row (differences must be new)
    # 2. Previously assigned elements in other rows (differences must be new)
    # 3. Future elements in the same row? No, we don't know their values yet.
    #    But we can check if the remaining space allows for valid assignments.
    #    This is hard.
    
    # Let's try a different angle: 
    # Since the number of rows (4) and k (7) are small, the total variables are 28.
    # The domain size is bounded by M.
    # If M is around 134, domain size is ~134.
    # 134^28 is huge. Backtracking will fail without very strong heuristics.
    
    # However, the constraints are very tight. 
    # Each new element eliminates many values.
    
    # Let's implement a recursive backtracking solver with memoization? 
    # State space is too large for memoization.
    
    # Let's use "Iterated Local Search" but with a deterministic repair step? 
    # No, the directive says "use backtracking with forward checking... rather than relying on stochastic optimization".
    
    # Okay, I will implement a backtracking solver. To make it work within 120s,
    # I will use a "Limited Discrepancy Search" or "Depth First Search with Restarting".
    # Actually, for small n_rows and k, a well-pruned DFS might work if we start with a good M.
    
    # Strategy:
    # 1. Calculate lower bound M_lb = total_diffs_needed.
    # 2. Try M = M_lb, M_lb + 1, ...
    # 3. For each M, run DFS.
    # 4. Use a time limit per M. If it fails, increment M.
    # 5. If we find a solution, return it.
    # 6. If time runs out, return the best valid solution found (which might be from a higher M if we found one, or a greedy one if we didn't find any valid one yet).
    
    # Wait, if we find a solution for M=134, that's likely the answer.
    # The greedy solution has scope ~? 
    # Let's check the greedy scope for (4,7).
    # Greedy:
    # Row 0: 0, 1, 2, 4, 7, 11, 16, 22 (differences: 1,2,4,7,11,16,22, 1,3,6,10,15,21, 2,5,9,14,20, 3,8,13,19, 4,10,16, 5,11,17, 6,12, 7,13, 8, 9)
    # This greedy might have a high scope.
    
    # Let's implement the backtracking solver.
    
    def solve_for_scope(M, time_limit):
        """
        Attempts to find a valid set of rows with max entry <= M.
        Uses backtracking.
        Returns list of rows if found, None otherwise.
        """
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        used_diffs = set()
        
        # We will fill rows one by one.
        # For each row, we fill elements 1..k.
        
        # To improve efficiency, we can randomize the order of rows?
        # No, symmetry breaking: Row 0 is fixed to start.
        # But we can randomize the order of filling elements within a row?
        # No, they must be increasing.
        
        # Let's use a recursive function.
        # To avoid deep recursion issues and allow time checking, we use a stack or careful recursion.
        # Python recursion limit might be an issue if depth is high, but depth is 28, so it's fine.
        
        def backtrack(r, c):
            if time.time() > time_limit:
                return False
            
            if r == n_rows:
                return True
            
            if c == k + 1:
                # Move to next row
                return backtrack(r + 1, 1)
            
            # Determine range for rows[r][c]
            lower = rows[r][c-1] + 1
            upper = M
            
            # Optimization: If c < k, the next element must be at least lower + 1.
            # So upper can be reduced if there isn't enough space for the rest.
            # Remaining elements in this row: k - c
            # Min space needed: (k - c) * 1 (if we pick consecutive integers)
            # So rows[r][c] <= M - (k - c)
            upper = min(upper, M - (k - c))
            
            if lower > upper:
                return False
            
            # Try values in [lower, upper]
            # Heuristic: Try smaller values first to keep scope low?
            # Or random order to escape local minima? 
            # For backtracking, deterministic order is standard. 
            # But to avoid getting stuck in a bad branch, we might randomize slightly.
            # Let's try deterministic first (increasing).
            
            for val in range(lower, upper + 1):
                # Check constraints
                valid = True
                
                # Check differences with previous elements in this row
                for j in range(c):
                    d = val - rows[r][j]
                    if d in used_diffs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check differences with elements in previous rows
                # Note: We have fully constructed rows 0..r-1.
                # So we must check all differences between val and any element in rows 0..r-1.
                for i in range(r):
                    for j in range(k + 1):
                        d = abs(val - rows[i][j])
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Commit
                rows[r][c] = val
                new_diffs = set()
                for j in range(c):
                    new_diffs.add(val - rows[r][j])
                for i in range(r):
                    for j in range(k + 1):
                        new_diffs.add(abs(val - rows[i][j]))
                
                used_diffs.update(new_diffs)
                
                if backtrack(r, c + 1):
                    return True
                
                # Uncommit
                used_diffs.difference_update(new_diffs)
                rows[r][c] = 0
            
            return False
        
        return backtrack(0, 1)

    # Greedy fallback
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    # Calculate lower bound for scope
    # Total diffs = n_rows * k * (k+1) / 2
    total_diffs = n_rows * k * (k + 1) // 2
    # The maximum difference is at most scope.
    # The number of distinct differences available in [1, scope] is scope.
    # So scope >= total_diffs.
    M_lb = total_diffs
    
    # We will try scopes from M_lb upwards.
    # To avoid wasting time on impossible small scopes, we can start a bit higher.
    # But M_lb is a hard bound.
    
    # We need to manage time.
    # If we spend too long on one M, we might miss the deadline.
    # Let's allocate a small time budget per M, increasing as M increases?
    # Or just run until deadline.
    
    current_M = M_lb
    # We might want to randomize the search order of values within the backtracking
    # to improve the chance of finding a solution if one exists.
    # The current backtracking is deterministic.
    # Let's add a randomization option to the backtracking.
    
    def solve_for_scope_randomized(M, time_limit):
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        used_diffs = set()
        
        def backtrack(r, c):
            if time.time() > time_limit:
                return False
            
            if r == n_rows:
                return True
            
            if c == k + 1:
                return backtrack(r + 1, 1)
            
            lower = rows[r][c-1] + 1
            upper = min(M, M - (k - c))
            
            if lower > upper:
                return False
            
            # Generate candidates
            candidates = list(range(lower, upper + 1))
            random.shuffle(candidates)
            
            for val in candidates:
                valid = True
                for j in range(c):
                    d = val - rows[r][j]
                    if d in used_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                for i in range(r):
                    for j in range(k + 1):
                        d = abs(val - rows[i][j])
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                rows[r][c] = val
                new_diffs = set()
                for j in range(c):
                    new_diffs.add(val - rows[r][j])
                for i in range(r):
                    for j in range(k + 1):
                        new_diffs.add(abs(val - rows[i][j]))
                used_diffs.update(new_diffs)
                
                if backtrack(r, c + 1):
                    return True
                
                used_diffs.difference_update(new_diffs)
                rows[r][c] = 0
            
            return False
        
        return backtrack(0, 1)

    # Try to improve on the greedy solution
    # Start from a scope slightly below the greedy scope?
    # No, greedy scope is an upper bound.
    # We want to find a smaller scope.
    # So we try M from M_lb to best_scope - 1.
    
    # To be efficient, we should try the most promising M first.
    # M_lb is the smallest possible.
    
    # However, backtracking might be slow.
    # Let's use a time limit for each M.
    # If we don't find a solution for M in time_limit, we move to M+1.
    
    # But if M is close to the solution, it might take longer.
    # Let's just run the randomized backtracking for increasing M until deadline.
    
    # Optimization: If we find a solution for M, we can stop.
    # We want the smallest M.
    
    # Let's iterate M from M_lb to best_scope - 1.
    # For each M, give it a portion of the remaining time.
    
    remaining_time = deadline - time.time()
    num_Ms_to_try = best_scope - M_lb
    if num_Ms_to_try <= 0:
        return best_rows
    
    time_per_M = remaining_time / num_Ms_to_try
    # Cap time_per_M to avoid spending too long on early, likely impossible M values
    # But M_lb is the theoretical minimum.
    
    for M in range(M_lb, best_scope):
        if time.time() > deadline - 1:
            break
            
        # Adjust time limit for this M
        # Give more time to M values closer to the current best?
        # Or just uniform.
        t_limit = time.time() + min(time_per_M, 5.0) # Cap at 5s per M to ensure progress
        
        sol = solve_for_scope_randomized(M, t_limit)
        
        if sol:
            # Check if it's valid (should be by construction)
            # Update best
            best_rows = sol
            best_scope = M
            # Since we are iterating upwards from the minimum, 
            # the first valid M is the optimal M for this search strategy.
            # We can return immediately? 
            # Yes, because we are looking for the minimum scope.
            return best_rows

    return best_rows
