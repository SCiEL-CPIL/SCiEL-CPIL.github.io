"""
Strategy: CSP Backtracking with Forward Checking.
Instead of starting with a loose solution and tightening it (Local Search/SA),
we assume a target scope M. We then use backtracking to construct the rows.
We treat each row as a variable set. To improve efficiency, we construct rows 
sequentially but use a "hole-filling" heuristic: we don't just pick random numbers,
we try to place the most constrained elements first or use a heuristic that 
minimizes the domain of remaining choices.

Key Structural Change:
1. Targeted Scope Search: We iterate M from the lower bound upwards.
2. CSP Solver: For a fixed M, we try to find n_rows rows.
   - We use a recursive backtracker.
   - We maintain a global set of used differences.
   - Heuristic: When constructing a row, we choose the next element such that 
     it leaves the most options for the rest of the row (or vice versa, 
     most constrained first). Given the small size (k=5, 7 rows), 
     we can afford a slightly more expensive check.
   - To handle the "increasing integers" constraint, we pick k numbers from 1..M.
   - We use randomization in the order of trying candidates to avoid getting 
     stuck in the same local trap if the first deterministic path fails.
"""

import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return list of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))

def _scope(rows):
    return max(r[-1] for r in rows)

def _solve_csp(n_rows, k, max_scope, rng, time_limit):
    """
    Try to construct n_rows rows of k+1 integers (0 + k values from 1..max_scope)
    such that all differences are distinct.
    Returns list of rows if found, None otherwise.
    """
    num_elements = k + 1
    # Total differences per row
    diffs_per_row = num_elements * (num_elements - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # If max_scope is too small to even hold the differences (pigeonhole), fail fast
    # Max difference is max_scope. Min is 1.
    # We need total_diffs distinct integers in [1, max_scope].
    if total_diffs > max_scope:
        return None

    used_diffs = set()
    rows = []
    
    # We will construct rows one by one.
    # To increase success rate, we can try to construct the "hardest" rows first?
    # Or just standard backtracking.
    # Since rows are identical in constraints, order doesn't matter for validity,
    # but it might matter for finding a solution quickly if we fix some structure.
    # However, simple backtracking with randomization is robust here.
    
    def backtrack(row_idx, current_row, start_val, used_diffs_local):
        """
        Recursively build current_row.
        row_idx: index of the row being built (0 to n_rows-1)
        current_row: list of elements in current row so far (starts with [0])
        start_val: minimum value for the next element
        used_diffs_local: set of differences used in this row so far
        """
        if time.time() > time_limit:
            return False

        if len(current_row) == num_elements:
            # Row complete. Check if any new conflicts with global used_diffs
            # Actually, we should check conflicts as we add elements to prune earlier.
            # But for simplicity of the "local" check, let's verify now.
            # If valid, add to rows and recurse to next row.
            rows.append(current_row[:])
            
            # Update global used_diffs
            row_diffs = _row_diffs(current_row)
            # Check if any of these are in global used_diffs
            # Wait, the backtracker should maintain global state or pass it down.
            # Passing a set down is expensive due to copying. 
            # Better approach: maintain global used_diffs, and undo on backtrack.
            
            # Let's refactor to use global state with undo.
            # This function structure is a bit rigid for that. 
            # Let's stick to a simpler iterative deepening or a different CSP formulation.
            
            # Actually, for k=5, n=7, the search space is small enough 
            # that we can just use a simple randomized backtracking that 
            # picks a full row candidate and checks it.
            # But that's what the previous program did (greedy).
            
            # The "CSP" advantage is pruning *within* the row construction.
            # Let's implement a proper backtracker that builds the row element by element.
            
            # This specific recursive call structure is flawed for global state management 
            # without significant overhead. 
            # Let's use a different approach: 
            # Generate a candidate row using a local backtracker that ensures 
            # internal distinctness and no conflict with global used_diffs.
            # Then add it. If we can't fill all rows, fail.
            
            # This is effectively what _greedy_row did, but without the "random sample" 
            # and instead using backtracking to find *one* valid row.
            # The previous program used random sampling which can miss valid rows 
            # that require specific non-random placements.
            
            # Let's rewrite the core logic to be a true backtracker for the whole set.
            return True # Placeholder, logic needs restructuring

    # --- RESTRUCTURED APPROACH ---
    # We will use a backtracking search that builds rows one by one.
    # For each row, we use a helper that finds a valid row given the current used_diffs.
    # If no valid row exists for the current used_diffs, we backtrack (change previous row).
    
    # To make this efficient, the helper "find_row" should be smart.
    # Instead of random sampling, it will use backtracking to fill the row.
    
    def find_row(used_diffs_global, max_scope, rng):
        """
        Find a row of length k+1 starting with 0, elements in [1, max_scope],
        such that all internal diffs are distinct and not in used_diffs_global.
        Returns row or None.
        Uses backtracking.
        """
        row = [0]
        
        def fill(pos, last_val):
            if time.time() > time_limit:
                return False
            if pos == num_elements:
                return True
            
            # Try values from last_val+1 to max_scope
            # Heuristic: Try values that create "small" differences first? 
            # Or random order? Random order is better for escaping local minima in the row space.
            candidates = list(range(last_val + 1, max_scope + 1))
            rng.shuffle(candidates)
            
            for val in candidates:
                # Check internal diffs with previous elements
                valid = True
                local_diffs = []
                for i in range(len(row)):
                    d = val - row[i]
                    if d in local_diffs:
                        valid = False
                        break
                    if d in used_diffs_global:
                        valid = False
                        break
                    local_diffs.append(d)
                
                if valid:
                    row.append(val)
                    # Recurse
                    if fill(pos + 1, val):
                        return True
                    row.pop()
            return False
        
        if fill(1, 0):
            return row
        return None

    # Main backtracking loop for rows
    # We don't need to backtrack across rows if we use a "commit" strategy,
    # but if row 1 blocks row 2, we might need to change row 1.
    # Given the small size, we can afford a limited backtracking depth or 
    # simply restart the whole process with a different random seed if we get stuck.
    # However, a true CSP solver would backtrack.
    
    # Let's implement a simple iterative deepening on the "row index"
    # Actually, since the problem is small (7 rows), we can just try to build 
    # all 7 rows sequentially. If we fail to build the last row, we can 
    # try restarting the whole thing with a different seed.
    # This is "Randomized Restart" of the greedy/CSP hybrid.
    
    # To be more robust, we can implement a limited backtracker:
    # If we fail at row i, we can try to modify row i-1.
    # But modifying a row requires finding a *different* valid row for i-1.
    
    # Let's stick to: 
    # 1. Try to build all rows sequentially using find_row.
    # 2. If any row fails, return None.
    # 3. The caller (construct) will restart with a new seed.
    
    current_used = set()
    current_rows = []
    
    for i in range(n_rows):
        # To avoid getting stuck in a bad local optimum for the row construction,
        # we can shuffle the order of rows? No, rows are identical.
        # We can shuffle the "target" values?
        
        row = find_row(current_used, max_scope, rng)
        if row is None:
            return None
        current_rows.append(row)
        # Update used diffs
        current_used.update(_row_diffs(row))
        
    return current_rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs

    # Counting bound: we need total_diffs_needed distinct differences in [1, M]
    # So M must be at least total_diffs_needed.
    # Also, the max difference is M, min is 1.
    # A tighter bound: The sum of the largest n differences?
    # Simple bound: M >= total_diffs_needed.
    # For (7,5), total_diffs = 7 * 15 = 105. So M >= 105.
    # Known best is 112.
    
    lower_bound = total_diffs_needed
    # Add a small buffer to lower bound to account for structural constraints
    initial_scope = lower_bound
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # We iterate scope upwards from lower_bound
    # We limit the number of restarts per scope to avoid wasting time on impossible scopes
    max_restarts_per_scope = 50
    
    # Upper bound for search
    # If we can't find a solution up to some limit, we stop.
    # Known best is 112. Let's search up to 150.
    max_search_scope = 150
    
    for scope in range(initial_scope, max_search_scope + 1):
        if time.time() > DEADLINE - 10:
            break
            
        # Try to find a solution with this scope
        found = False
        for restart in range(max_restarts_per_scope):
            # Use a different seed for each restart to explore different parts of the space
            rng_restart = random.Random(1000 + scope * 100 + restart)
            
            rows = _solve_csp(n_rows, k, scope, rng_restart, time_limit=DEADLINE - 5)
            if rows is not None:
                # Validate just in case
                if _is_valid(rows):
                    s = _scope(rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = rows
                    found = True
                    break # Found a solution for this scope, move to next (smaller? no, we go up)
                    # Wait, we are going up from lower_bound. 
                    # If we find a solution at scope S, that's the best for this loop iteration?
                    # No, we want the SMALLEST scope.
                    # So if we start at lower_bound and go up, the FIRST scope where we find a solution
                    # is the optimal (or near optimal) answer.
                    # So we can return immediately?
                    # Yes, because we are iterating from the theoretical minimum upwards.
                    # If we find a valid set at M, then M is the answer (unless there's a bug).
                    # However, due to search incompleteness, we might miss a solution at M 
                    # and find one at M+1. 
                    # To be safe, we should try a few restarts at M. If we fail, we go to M+1.
                    # If we succeed at M+1, we return M+1.
                    # This guarantees we return the smallest M for which we *found* a solution.
                    
        if found:
            # Since we iterate from low to high, this is the best found so far.
            # We can break here because any higher scope is worse.
            # Is it possible that a lower scope was skipped? No, we started at lower_bound.
            # So this is the minimal scope we found.
            break
            
    # If we didn't find anything (unlikely), we need a fallback
    if best_rows is None:
        # Fallback: Use a very large scope and greedy
        scope = 300
        for _ in range(100):
            rows = _solve_csp(n_rows, k, scope, rng, time_limit=DEADLINE - 5)
            if rows is not None and _is_valid(rows):
                best_rows = rows
                best_scope = _scope(rows)
                break
        if best_rows is None:
            # Absolute fallback: sequential construction
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            best_rows = rows
            best_scope = _scope(rows)

    return best_rows

# EVOLVE-BLOCK-END
