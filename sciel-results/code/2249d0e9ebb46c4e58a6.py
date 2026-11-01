import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    # The problem: Find n_rows sets of k+1 integers starting with 0.
    # Let the sets be R_1, ..., R_n.
    # Each R_i = {0, x_{i,1}, ..., x_{i,k}} with 0 < x_{i,1} < ... < x_{i,k}.
    # All differences |x - y| for x, y in R_i, x != y must be distinct across all i.
    # Minimize max(x_{i,k}).
    
    # Total number of differences per row: C(k+1, 2) = k(k+1)/2.
    # Total differences overall: n_rows * k(k+1)/2.
    # If the max element is M, the possible differences are 1..M.
    # So we need M >= n_rows * k(k+1)/2.
    # For n_rows=4, k=7: 4 * 28 = 112.
    # Best known is 134.
    
    # Strategy: CSP with Backtracking and Forward Checking.
    # We build the solution row by row.
    # State: 
    #   - used_diffs: set of all differences used so far.
    #   - rows: list of completed rows.
    #   - current_row: list of elements in the row currently being built.
    
    # We use a recursive backtracking search.
    # To make it fast, we use:
    # 1. Minimum Remaining Values (MRV) heuristic isn't directly applicable to value selection 
    #    in a simple sequence, but we can order the rows or use specific value heuristics.
    # 2. We will try to construct rows in a specific order. 
    #    Often, constructing "greedy" first and then filling gaps or using a more balanced approach works.
    # 3. We will use a "Limited Discrepancy Search" (LDS) or simple backtracking with a time limit.
    
    # Given the small size (4 rows, 8 elements), a well-guided backtracking search 
    # should be able to find a solution near the lower bound if we guide it well.
    
    # Let's define the search space.
    # We want to find rows such that the union of their difference sets is disjoint.
    
    # We will use a randomized backtracking search with restarts.
    # Each restart tries to build the solution from scratch.
    # We use a "soft" constraint approach initially? No, the task requires validity.
    # We must return a valid set. If we don't find one in time, we return the best valid one found?
    # Wait, the evaluator says "Valid when...". If invalid, score 0.
    # So we MUST return a valid solution.
    # The greedy init in the previous code was valid.
    # So we can start with a valid solution (from greedy) and then try to improve it?
    # Or we can just run the CSP solver until it finds a valid solution.
    # Since a valid solution always exists (e.g. very large numbers), we just need to find a good one.
    # The "best known" is 134. The counting bound is 112.
    # We aim to find a solution with scope < 134 if possible, or at least valid.
    
    # Let's implement a backtracking solver that tries to keep the scope small.
    # We will iterate on the target scope M.
    # Start M from the counting bound.
    # Try to find a solution with max element <= M.
    # If found, return it.
    # If not found within time limit for this M, increase M.
    
    # However, backtracking can be exponential.
    # We need strong pruning.
    
    # Pruning:
    # When adding a new element x to the current row, 
    # all differences x - y for y in current_row must not be in used_diffs.
    # Also, they must be distinct within the current row (guaranteed if x > y and we check against current row diffs).
    
    # Let's optimize the difference check.
    
    def greedy_init():
        """Generate a valid solution using simple greedy. This is our fallback."""
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
                        if (x - a) in used_diffs:
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

    # Get a baseline valid solution
    baseline = greedy_init()
    best_solution = baseline
    best_scope = max(max(r) for r in baseline)
    
    # We will try to improve upon the baseline using a CSP approach.
    # We'll use a randomized backtracking search.
    
    def is_valid_solution(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    # Backtracking search function
    # We try to build rows one by one.
    # To break symmetry, we can fix the first row to be a specific "greedy" row 
    # or try to build all rows simultaneously?
    # Building row by row is simpler.
    
    # Let's use a "randomized greedy with backtracking" for each row.
    
    def solve_csp(target_scope):
        """
        Try to find a solution with max element <= target_scope.
        Returns list of rows or None.
        """
        used_diffs = set()
        rows = []
        
        # We will build rows one by one.
        # To improve success rate, we can shuffle the order of rows?
        # No, rows are identical in constraint.
        
        # Heuristic: Try to pack the differences densely.
        
        def backtrack(row_idx, current_row, current_used_diffs_in_row):
            if row_idx == n_rows:
                return True
            
            # If we are starting a new row
            if not current_row:
                current_row = [0]
                current_used_diffs_in_row = set()
            
            if len(current_row) == k + 1:
                # Row complete
                # Add to rows
                rows.append(current_row)
                # Update global used_diffs
                # Note: current_used_diffs_in_row contains all diffs for this row
                
                # Recurse to next row
                if backtrack(row_idx + 1, [], set()):
                    return True
                
                # Backtrack
                rows.pop()
                # Remove diffs from used_diffs? 
                # We are using a set passed by reference or copying?
                # For efficiency, we should use a set and add/remove.
                # But Python sets are cheap to copy for small sizes? 
                # k=7, 28 diffs. 4 rows. 112 diffs total.
                # Copying a set of 112 ints is fast.
                # Let's pass used_diffs as a parameter (copy on write style for simplicity/safety)
                # Or use a global set and add/remove.
                # Let's use add/remove for speed.
                
                # Wait, the function signature above doesn't show used_diffs being managed.
                # Let's rewrite backtrack to manage a global used_diffs set.
                pass
            
            # Find next element
            # Current max in row
            last_val = current_row[-1]
            
            # We need to pick x > last_val such that x <= target_scope
            # And x - y not in used_diffs (global) for all y in current_row
            # And x - y not in current_used_diffs_in_row for all y in current_row (internal distinctness)
            
            # Optimization: 
            # The set of forbidden values for x is:
            # F = { y + d | y in current_row, d in used_diffs } U { y + d | y in current_row, d in current_used_diffs_in_row }
            # We want the smallest x > last_val not in F and x <= target_scope.
            
            # Actually, we can just iterate x from last_val + 1 to target_scope.
            # Given target_scope is ~112-134, and row length 8, this is fast.
            
            for x in range(last_val + 1, target_scope + 1):
                # Check constraints
                valid = True
                for y in current_row:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    if d in current_used_diffs_in_row:
                        valid = False
                        break
                
                if valid:
                    # Commit
                    new_row = current_row + [x]
                    new_internal_diffs = set(current_used_diffs_in_row)
                    new_internal_diffs.update(x - y for y in current_row)
                    
                    # Add to global used_diffs
                    for y in current_row:
                        used_diffs.add(x - y)
                    
                    # Recurse
                    if backtrack(row_idx, new_row, new_internal_diffs):
                        return True
                    
                    # Uncommit
                    for y in current_row:
                        used_diffs.remove(x - y)
            
            return False
        
        # Start backtracking
        # We start with row 0, empty row (will be initialized to [0] in logic above? 
        # My logic above was a bit messy. Let's clean it up.
        
        used_diffs = set()
        rows = []
        
        def _backtrack(row_idx, current_row, internal_diffs):
            if row_idx == n_rows:
                return True
            
            if not current_row:
                current_row = [0]
                internal_diffs = set()
            
            if len(current_row) == k + 1:
                # Row is complete
                rows.append(current_row)
                # Note: internal_diffs are already in used_diffs from the step-by-step addition
                if _backtrack(row_idx + 1, [], set()):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Heuristic: Try values in a randomized order?
            # Or smallest first?
            # Smallest first tends to pack tighter, which is good for minimizing scope.
            # But it might lead to dead ends sooner.
            # Let's try smallest first, but with a "jump" if stuck?
            # For now, smallest first.
            
            for x in range(last_val + 1, target_scope + 1):
                valid = True
                for y in current_row:
                    d = x - y
                    if d in used_diffs or d in internal_diffs:
                        valid = False
                        break
                
                if valid:
                    # Commit
                    new_row = current_row + [x]
                    new_internal = set(internal_diffs)
                    for y in current_row:
                        used_diffs.add(x - y)
                        new_internal.add(x - y)
                    
                    if _backtrack(row_idx, new_row, new_internal):
                        return True
                    
                    # Uncommit
                    for y in current_row:
                        used_diffs.remove(x - y)
            
            return False
        
        # Start
        if _backtrack(0, [], set()):
            return rows
        return None

    # Main loop: Try increasing target scopes
    # Start from counting bound
    counting_bound = n_rows * k * (k + 1) // 2
    
    # We know baseline scope. We want to beat it.
    # Let's try from counting_bound up to best_scope - 1.
    # If we find a solution, update best_scope.
    
    # To avoid getting stuck in one bad configuration, we can use randomized restarts 
    # or randomize the order of x selection slightly.
    # For simplicity and determinism in the first pass, we'll use pure backtracking.
    # If it's too slow, we'll add randomization.
    
    # Given the small size, pure backtracking might work.
    # However, the search space is huge.
    # We need a better heuristic.
    
    # Alternative: Use the "Best Fit" idea but with strict backtracking.
    # The previous code's "construct_best_row" was doing a greedy search.
    # Let's stick to the CSP solver but add a time check inside the backtrack.
    
    global_deadline = deadline
    
    def solve_csp_timed(target_scope):
        used_diffs = set()
        rows = []
        
        def _backtrack(row_idx, current_row, internal_diffs):
            if time.time() > global_deadline:
                raise TimeoutError
            
            if row_idx == n_rows:
                return True
            
            if not current_row:
                current_row = [0]
                internal_diffs = set()
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if _backtrack(row_idx + 1, [], set()):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Try x from last_val+1 to target_scope
            # To add some randomness, we could shuffle the range, but that breaks the "smallest first" packing.
            # Let's stick to smallest first for now.
            
            for x in range(last_val + 1, target_scope + 1):
                valid = True
                for y in current_row:
                    d = x - y
                    if d in used_diffs or d in internal_diffs:
                        valid = False
                        break
                
                if valid:
                    new_row = current_row + [x]
                    new_internal = set(internal_diffs)
                    for y in current_row:
                        used_diffs.add(x - y)
                        new_internal.add(x - y)
                    
                    if _backtrack(row_idx, new_row, new_internal):
                        return True
                    
                    for y in current_row:
                        used_diffs.remove(x - y)
            
            return False
        
        try:
            if _backtrack(0, [], set()):
                return rows
        except TimeoutError:
            pass
        return None

    # Try to find a better solution
    # We will try scopes from counting_bound to best_scope - 1
    # We might not find one quickly.
    # To improve odds, we can randomize the "start" of the search or use a different heuristic.
    
    # Let's try a few randomized restarts of the CSP solver with a fixed target scope.
    # If we can't beat the baseline, we return the baseline.
    
    # Since the baseline is valid, we are safe.
    # The goal is to improve the score.
    
    # Let's try to find a solution with scope = counting_bound.
    # If that fails, try counting_bound + 1, etc.
    # We limit the time for each target scope to avoid getting stuck.
    
    # Actually, the backtracking is depth-first. It might explore a huge branch.
    # Let's add a "node limit" or "time slice" per restart.
    
    # For now, let's just run the CSP solver for a few seconds and see if it finds anything better.
    # If it times out, we return the baseline.
    
    # To make the CSP solver more effective, we can use "failure-directed search" 
    # or simply rely on the fact that the counting bound is tight.
    
    # Let's try a few random permutations of the row construction order?
    # No, rows are symmetric.
    
    # Let's try randomizing the value selection order in the CSP solver slightly.
    # Instead of strictly increasing x, we can try x in a random order 
    # within a window? No, that breaks the monotonicity of the row.
    # The row elements MUST be increasing.
    
    # So the only freedom is which x to pick at each step.
    # "Smallest first" is the standard heuristic for packing.
    
    # Let's just run the CSP solver with a time limit of 110s.
    # If it finds a solution with scope < best_scope, we use it.
    # Otherwise, we use the baseline.
    
    # To increase the chance of finding a solution near the bound, 
    # we can try multiple restarts with slightly different "greedy" biases.
    
    # But for this specific task (4,7), the structure is very constrained.
    # The known solution 134 is likely the minimum or close to it.
    # The counting bound is 112.
    
    # Let's try to solve for scope = 112, 113, ... up to best_scope.
    
    # We will use a simple loop.
    
    current_best = best_solution
    current_best_scope = best_scope
    
    # Try to improve
    # We will try scopes from counting_bound to current_best_scope - 1
    
    # To avoid hanging on a single scope, we allocate a time slice to each.
    # Total time available: ~110s.
    # Number of scopes to try: current_best_scope - counting_bound.
    # If best_scope is 134 and bound is 112, that's 22 scopes.
    # 110s / 22 ~ 5s per scope.
    
    scopes_to_try = range(counting_bound, current_best_scope)
    
    for scope in scopes_to_try:
        if time.time() > deadline - 5:
            break
        
        # Allocate time slice
        slice_deadline = time.time() + 4.0
        
        # Modify global_deadline temporarily? 
        # Better to pass deadline to solver.
        
        # Let's create a solver that respects a local deadline.
        
        def solve_with_deadline(target_scope, dl):
            used_diffs = set()
            rows = []
            
            def _backtrack(row_idx, current_row, internal_diffs):
                if time.time() > dl:
                    raise TimeoutError
                
                if row_idx == n_rows:
                    return True
                
                if not current_row:
                    current_row = [0]
                    internal_diffs = set()
                
                if len(current_row) == k + 1:
                    rows.append(current_row)
                    if _backtrack(row_idx + 1, [], set()):
                        return True
                    rows.pop()
                    return False
                
                last_val = current_row[-1]
                
                for x in range(last_val + 1, target_scope + 1):
                    valid = True
                    for y in current_row:
                        d = x - y
                        if d in used_diffs or d in internal_diffs:
                            valid = False
                            break
                    
                    if valid:
                        new_row = current_row + [x]
                        new_internal = set(internal_diffs)
                        for y in current_row:
                            used_diffs.add(x - y)
                            new_internal.add(x - y)
                        
                        if _backtrack(row_idx, new_row, new_internal):
                            return True
                        
                        for y in current_row:
                            used_diffs.remove(x - y)
                
                return False
            
            try:
                if _backtrack(0, [], set()):
                    return rows
            except TimeoutError:
                pass
            return None

        sol = solve_with_deadline(scope, slice_deadline)
        if sol:
            # Verify solution
            if is_valid_solution(sol):
                new_scope = get_scope(sol)
                if new_scope < current_best_scope:
                    current_best = sol
                    current_best_scope = new_scope
                    # If we found a solution at the counting bound, we can stop
                    if scope == counting_bound:
                        break

    return current_best

# Test locally
if __name__ == "__main__":
    res = construct(4, 7)
    print(res)
    print("Scope:", max(max(r) for r in res))
