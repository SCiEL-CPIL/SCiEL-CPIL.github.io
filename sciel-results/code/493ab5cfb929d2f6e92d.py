import time
import itertools
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    # Total number of differences per row
    diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound for M: M(M+1)/2 >= total_diffs_needed
    # M^2 + M - 2*total_diffs_needed >= 0
    # M >= (-1 + sqrt(1 + 8*total_diffs_needed)) / 2
    import math
    M_min = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2))
    
    best_solution = None
    best_M = float('inf')
    
    # We will try M values starting from M_min
    # We don't know the upper bound, but for (7,5) it's 112.
    # We can go up to, say, 200 or until time runs out.
    
    # To make the search faster, we can use a randomized restart approach if it gets stuck,
    # but for small M, deterministic search is usually fine.
    # However, to avoid getting stuck in a deep dead-end for a specific M,
    # we can limit the depth or use heuristics.
    
    # Let's implement the backtracking search for a fixed M.
    
    def solve_for_M(M):
        """
        Returns a list of rows if a valid configuration exists with max mark <= M, else None.
        """
        used_diffs = set()
        rows = []
        
        # Pre-calculate max differences available? No, dynamic.
        
        def backtrack(row_idx, current_row, last_mark):
            nonlocal used_diffs, rows, best_solution, best_M, start_time, deadline
            
            # Time check
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                # Found a solution
                # We copy the rows
                best_solution = [r[:] for r in rows]
                best_M = M
                return True
            
            # Heuristic: Check if enough differences remain
            # Remaining rows: n_rows - row_idx
            # Diffs needed: (n_rows - row_idx) * diffs_per_row
            # Diffs used: len(used_diffs)
            # Total diffs in [1, M]: M*(M+1)//2
            # If len(used_diffs) + needed > total_available: prune
            # Note: This is a necessary condition, not sufficient.
            remaining_rows = n_rows - row_idx
            needed = remaining_rows * diffs_per_row
            total_avail = M * (M + 1) // 2
            if len(used_diffs) + needed > total_avail:
                return False
            
            # Start a new row
            new_row = [0]
            # The last mark in the new row so far is 0
            # We need to add k more marks.
            
            def extend_row(pos, last_in_row):
                nonlocal used_diffs, rows, best_solution, best_M, start_time, deadline
                
                if time.time() > deadline:
                    return False
                
                if pos == k + 1:
                    # Row complete
                    rows.append(new_row)
                    if backtrack(row_idx + 1, [], 0):
                        return True
                    rows.pop()
                    return False
                
                # Try candidates for the next mark
                # Candidate range: last_in_row + 1 to M
                # Optimization: We can limit the upper bound of the next mark.
                # The remaining marks in this row must fit between last_in_row and M.
                # We need (k - pos) more marks.
                # So the next mark cannot be greater than M - (k - pos).
                max_candidate = M - (k - pos)
                
                # Iterate candidates
                # To find the "best" (smallest max), we iterate from smallest to largest.
                for x in range(last_in_row + 1, max_candidate + 1):
                    # Check differences
                    new_diffs = set()
                    conflict = False
                    for a in new_row:
                        d = x - a
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    
                    if conflict:
                        continue
                    
                    # Also check if new_diffs have internal conflicts? 
                    # No, since x > a, and a are distinct, x-a are distinct.
                    
                    # Add to row and used
                    new_row.append(x)
                    used_diffs.update(new_diffs)
                    
                    if extend_row(pos + 1, x):
                        return True
                    
                    # Undo
                    new_row.pop()
                    used_diffs.difference_update(new_diffs)
                
                return False
            
            # Start extending the row
            return extend_row(1, 0)
        
        return backtrack(0, [], 0)
    
    # Search for the minimum M
    # We try M from M_min upwards.
    # To avoid spending too much time on a single M that fails, we could add a node limit,
    # but for small problems, it's usually fast.
    # However, if M is too small, it fails quickly. If M is just above the minimum, it might take longer.
    
    M = M_min
    while M < 200: # Arbitrary upper bound, will be cut by time
        if time.time() > deadline:
            break
        
        # Reset state for this M
        # Note: solve_for_M uses nonlocal variables from construct scope, 
        # but we need to be careful with state. 
        # The nested function modifies used_diffs and rows.
        # We need to ensure they are reset for each M.
        # Let's refactor solve_for_M to be self-contained or reset state properly.
        
        # Let's just call the search.
        # We need to clear used_diffs and rows before each M attempt.
        # Since they are nonlocal, we can just clear them.
        # But the nested functions capture them by reference.
        
        # Let's wrap the logic in a class or just reset the lists/sets.
        # used_diffs.clear()
        # rows.clear()
        
        # Actually, the nested functions in Python closures share the variable names.
        # So we can just clear the sets/lists.
        # But `best_solution` and `best_M` are updated inside.
        
        # Let's re-structure slightly to be safe.
        
        # We'll just run the search.
        # If it finds a solution, best_solution is set.
        
        # We need to clear used_diffs and rows before calling backtrack.
        # The `backtrack` function uses `rows` and `used_diffs`.
        
        # Let's move the state initialization into the loop.
        
        # Clear state
        # We need access to the state variables.
        # Let's define the state variables outside the loop but inside construct.
        
        # Refactor:
        # state_used = set()
        # state_rows = []
        
        # The nested functions should use these.
        
        # Let's rewrite the inner search to take M and use fresh state.
        
        def try_M(M_val):
            used = set()
            rows_list = []
            
            def bt(row_idx):
                if time.time() > deadline:
                    return False
                
                if row_idx == n_rows:
                    # Success
                    # We need to update the outer best_solution
                    # We can return the rows
                    return rows_list[:]
                
                # Pruning
                remaining_rows = n_rows - row_idx
                needed = remaining_rows * diffs_per_row
                total_avail = M_val * (M_val + 1) // 2
                if len(used) + needed > total_avail:
                    return None
                
                new_row = [0]
                
                def extend(pos, last_mark):
                    if time.time() > deadline:
                        return None
                    
                    if pos == k + 1:
                        rows_list.append(new_row)
                        res = bt(row_idx + 1)
                        if res is not None:
                            return res
                        rows_list.pop()
                        return None
                    
                    max_cand = M_val - (k - pos)
                    
                    for x in range(last_mark + 1, max_cand + 1):
                        new_diffs = set()
                        conflict = False
                        for a in new_row:
                            d = x - a
                            if d in used:
                                conflict = True
                                break
                            new_diffs.add(d)
                        
                        if conflict:
                            continue
                        
                        new_row.append(x)
                        used.update(new_diffs)
                        
                        res = extend(pos + 1, x)
                        if res is not None:
                            return res
                        
                        new_row.pop()
                        used.difference_update(new_diffs)
                    
                    return None
                
                return extend(1, 0)
            
            return bt(0)
        
        result = try_M(M)
        if result is not None:
            return result
        
        M += 1
        
    # If we didn't find a solution in the loop, return the best found so far?
    # Or if no solution found, we must return *something* valid.
    # The greedy seed always returns a valid solution.
    # We should probably fall back to a greedy construction if we run out of time or search space.
    
    # Fallback: Greedy construction (similar to seed but maybe slightly better ordered)
    # Or just return the seed's logic result to ensure validity.
    # The seed logic:
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
    return rows

# EVOLVE-BLOCK-END
