import time
import sys

def construct(n_rows, k):
    # k is the number of increments, so row length is k+1
    # Based on prompt "7 rows of 6 increasing integers", k should be 5.
    # If k is passed as 6, row length is 7.
    # The prompt says "Interface: construct(n_rows, k)... Each row is a list of k + 1 integers".
    # And "Task: Difference triangle set (7,5)".
    # I will assume the evaluator passes k=5 for the 7x6 case.
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Lower bound for M
    # Total diffs = n_rows * (k+1)*k/2
    # M must be at least total_diffs
    num_marks = k + 1
    diffs_per_row = num_marks * (num_marks - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # We search for the smallest M
    # Start from total_diffs
    # We can use a binary search or linear scan. Linear scan from LB is safe.
    # To speed up, we can start from a known good upper bound if we have one, 
    # but we don't. We'll start from LB.
    
    # To improve speed, we can try to find a solution for M, M+1, etc.
    # We need to store the best solution found (lowest M)
    best_solution = None
    best_M = float('inf')
    
    # Symmetry breaking: Rows are indistinguishable.
    # We can enforce that the rows are sorted lexicographically.
    # This means row[i] <= row[i+1] lexicographically.
    
    def solve_rows(current_row_idx, current_row, used_diffs, rows_so_far, M):
        """
        current_row_idx: index of the row currently being built
        current_row: list of marks in the current row
        used_diffs: set of all differences used so far
        rows_so_far: list of completed rows
        M: maximum allowed value for any mark
        """
        if time.time() > deadline:
            return None
        
        # Check if all rows are built
        if current_row_idx == n_rows:
            return rows_so_far.copy()
        
        # If we are starting a new row, we need to ensure it's >= the previous row (lexicographic)
        # To simplify, we just build rows one by one.
        # The symmetry breaking "row i <= row i+1" is tricky to enforce during construction.
        # Alternative: Just build any set of rows. The search space is larger, but with M small, it's manageable.
        # Given M ~ 110, the number of possible rows is limited.
        
        # Let's stick to building rows sequentially without strict lexicographic ordering first, 
        # but we can add a check: if current_row_idx > 0, the final row must be >= rows_so_far[-1].
        # Actually, it's easier to just let it find any combination.
        
        # If current row is complete
        if len(current_row) == num_marks:
            # Check if this row is valid wrt symmetry? 
            # If we enforce rows_so_far[-1] <= current_row (lexicographically), we cut space.
            if current_row_idx > 0 and current_row < rows_so_far[-1]:
                return None
            
            # Recurse to next row
            return solve_rows(current_row_idx + 1, [], used_diffs, rows_so_far + [current_row], M)
        
        # Determine the last mark in current row
        last_mark = current_row[-1] if current_row else 0
        
        # Try next marks
        # The next mark must be > last_mark
        # Also, the mark must be <= M
        # Optimization: We can limit the range of the next mark.
        # The smallest possible next mark is last_mark + 1.
        # The largest possible next mark is M.
        
        # Heuristic: Try smaller values first to keep M small? 
        # But M is fixed in this search. We are checking if a solution exists for this M.
        # So we just need to find ANY solution.
        
        # To speed up, we can look at the "gap" to the next required difference.
        
        for x in range(last_mark + 1, M + 1):
            # Check differences
            new_diffs = set()
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Additional Pruning: 
            # If we are at the last position (len(current_row) == num_marks - 1), 
            # we just need to check if x fits.
            
            # Forward Checking: 
            # Are there enough unused differences left for the rest of the rows?
            # Total remaining diffs needed = (n_rows - current_row_idx - 1) * diffs_per_row + (diffs_per_row - len(new_diffs) - (len(current_row)))
            # This is a weak bound.
            
            # Stronger Pruning: 
            # The maximum difference in the final set must be <= M.
            # This is already enforced by x <= M.
            
            # Let's try the recursive call
            result = solve_rows(
                current_row_idx, 
                current_row + [x], 
                used_diffs | new_diffs, 
                rows_so_far, 
                M
            )
            if result is not None:
                return result
        
        return None

    # Main loop
    # We start from the lower bound
    # To avoid getting stuck in a very hard instance at the LB, we might want to 
    # interleave or just go up.
    
    # Note: The problem is small enough that linear scan from LB is fine.
    # LB = 105 for k=5.
    
    # However, if k is different, LB changes.
    # Let's add a quick greedy to get an upper bound first?
    # Greedy is fast.
    
    def greedy_solve():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < num_marks:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        break
                    x += 1
            rows.append(row)
        return rows

    # Get an initial upper bound
    init_rows = greedy_solve()
    if init_rows:
        best_M = max(max(r) for r in init_rows)
        best_solution = init_rows
    else:
        best_M = total_diffs + 100 # Fallback
    
    # Now search downwards? Or upwards from LB?
    # Searching upwards from LB (105) is safer because we know a solution exists at best_M.
    # We can binary search between LB and best_M.
    
    low = total_diffs
    high = best_M
    
    # We want the smallest M.
    # Binary search for the first M that has a solution.
    
    # But backtracking is exponential. 
    # If we pick a small M, it might fail quickly (pruning is effective when M is small because few choices).
    # If we pick a large M, it might find a solution quickly but it's not minimal.
    # We want minimal.
    
    # Strategy: 
    # 1. Try M = low. If solution, return.
    # 2. If not, M = low + 1.
    # ...
    # This is linear.
    # Given the gap between 105 and 112 is small, linear scan is fine.
    
    # Optimization: 
    # Use the greedy result as a "hint" or just the upper bound.
    
    current_M = low
    
    # We can also randomize the order of rows to break symmetry more effectively?
    # No, lexicographic sorting of rows is a good static symmetry break.
    
    while current_M <= high:
        if time.time() > deadline:
            break
            
        # Clear cache/state for this M
        # We use a fresh used_diffs set for each M attempt
        
        # To speed up the search for a specific M, we can use a more aggressive heuristic:
        # Order the rows? No, we build them.
        
        # Let's run the solver
        # We need to reset the recursion state.
        # The solver function above uses recursion.
        
        # One issue: The solver above doesn't use the "rows_so_far" for pruning except for the lex order.
        # We can add a check: if the current row's potential max difference exceeds M? 
        # No, x is bounded by M.
        
        # Let's try to solve for current_M
        # We need to be careful with the recursion limit. 
        # Depth is small.
        
        # To make it faster, we can pass the `used_diffs` as a bitmask if M is small enough.
        # M ~ 112. 112 bits fits in a Python int easily.
        
        def solve_with_mask(current_row_idx, current_row, used_mask, rows_so_far, M):
            if time.time() > deadline:
                return None
            
            if current_row_idx == n_rows:
                return rows_so_far.copy()
            
            if len(current_row) == num_marks:
                if current_row_idx > 0 and current_row < rows_so_far[-1]:
                    return None
                return solve_with_mask(current_row_idx + 1, [], used_mask, rows_so_far + [current_row], M)
            
            last_mark = current_row[-1] if current_row else 0
            
            # Iterate x
            for x in range(last_mark + 1, M + 1):
                new_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if (used_mask & bit) or (new_mask & bit):
                        valid = False
                        break
                    new_mask |= bit
                
                if not valid:
                    continue
                
                # Pruning: 
                # If we are filling the last slot, just check.
                
                result = solve_with_mask(
                    current_row_idx, 
                    current_row + [x], 
                    used_mask | new_mask, 
                    rows_so_far, 
                    M
                )
                if result is not None:
                    return result
            
            return None

        solution = solve_with_mask(0, [], 0, [], current_M)
        
        if solution is not None:
            best_solution = solution
            best_M = current_M
            break # Found the minimum M
        
        current_M += 1
        
    # If we timed out or didn't find a better one, return the best we have (greedy)
    if best_solution is None:
        # Fallback: return a trivial valid set if possible, or empty?
        # The task requires a valid set.
        # Greedy always produces a valid set.
        best_solution = greedy_solve()
        
    return best_solution
