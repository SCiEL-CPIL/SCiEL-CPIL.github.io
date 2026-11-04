"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    # We'll use a bitset (Python int) to track which differences are used.
    # Bit d is set if difference d is already used.
    # We need differences up to some max M.

    best_rows = None
    best_max = float('inf')

    def solve_with_backtracking(seed, max_limit):
        """
        Depth-first backtracking with forward checking using bitsets.
        We build rows one entry at a time.
        
        State:
          - rows: list of lists, partially filled
          - used_diffs: integer bitset where bit d is 1 if diff d is used
          - current row index and column index
        
        We fill in row-major order: row 0 col 1, row 0 col 2, ..., row 0 col k,
        then row 1 col 1, etc.
        
        For each new value x at position (r, c):
          - x must be > rows[r][c-1]
          - For each i < c, the difference x - rows[r][i] must not be in used_diffs
          - Also, these new differences become "temporarily used" for the rest of this row
            but we need to be careful: differences within the same row are fine as long
            as they're distinct from ALL other differences (including within the same row).
        
        Actually, let me reconsider. The constraint is:
          All positive differences a_j - a_i (i < j) inside the rows are distinct over all rows.
        
        So if I'm building row r, and I've placed entries at positions 0, 1, ..., c-1,
        and I'm about to place entry c with value x, then:
          - For each i in 0..c-1: d = x - rows[r][i] must not be in used_diffs
          - AND d must not equal any difference already in this row (which is tracked by used_diffs
            if we add them as we go)
        
        Wait, used_diffs should track ALL differences placed so far (across all rows and within rows).
        So when we place a new entry, we check all new differences it creates against used_diffs,
        and if valid, we add them to used_diffs.
        
        This is a clean forward-checking approach.
        """
        rng = random.Random(seed)
        
        # rows[r][c] for r in 0..n_rows-1, c in 0..k
        # rows[r][0] = 0 for all r
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        
        # used_diffs bitset: bit d set means difference d is used
        # We need to handle differences up to max_limit
        # Python ints can be arbitrarily large, so this is fine
        used_diffs = 0  # no differences used initially
        
        # We'll use a recursive backtracking approach
        # Position: (row, col) where col goes from 1 to k (col 0 is fixed at 0)
        
        def backtrack(row_idx, col_idx):
            nonlocal used_diffs, best_rows, best_max
            
            if time.time() > deadline:
                return False
            
            # If we've filled all positions, we have a solution
            if row_idx == n_rows:
                # Check if this is better than our best
                max_entry = 0
                for r in range(n_rows):
                    if rows[r][k] > max_entry:
                        max_entry = rows[r][k]
                if max_entry < best_max:
                    best_max = max_entry
                    best_rows = [row[:] for row in rows]
                return True
            
            # Get the current row
            row = rows[row_idx]
            
            # The previous value (must be less than new value)
            prev_val = row[col_idx - 1]
            
            # Generate candidate values
            # We want to try values that are likely to work
            # Start from prev_val + 1 and go up to max_limit
            
            # Forward checking: for each candidate x, check if all new differences
            # (x - row[i] for i in 0..col_idx-1) are not in used_diffs
            
            # To make this more efficient, we can generate candidates in a smart order
            # Try smaller values first (to keep max small), but also allow some randomization
            
            # Generate candidate values
            candidates = []
            min_val = prev_val + 1
            max_val = max_limit
            
            if min_val > max_val:
                return False
            
            # We'll try values in increasing order, but with some randomization
            # to avoid always getting stuck in the same local minima
            
            # Strategy: try values in a range, prioritizing smaller ones
            # but also allowing some exploration
            
            # For efficiency, let's try a window of candidates
            # Start with small values, expand if needed
            
            # Try values from min_val up to some limit
            # We'll use a step to avoid checking every single value
            
            # Actually, let's just try all values from min_val to max_val
            # but with a timeout check
            
            for x in range(min_val, max_val + 1):
                if time.time() > deadline:
                    return False
                
                # Check if all new differences are valid
                valid = True
                new_diffs = 0
                for i in range(col_idx):
                    d = x - row[i]
                    if d <= 0:
                        valid = False
                        break
                    # Check if d is already used
                    if (used_diffs >> d) & 1:
                        valid = False
                        break
                    new_diffs |= (1 << d)
                
                if not valid:
                    continue
                
                # Also check that the new differences don't conflict with each other
                # (they can't, since they're all different values of d)
                
                # Place the value
                row[col_idx] = x
                used_diffs |= new_diffs
                
                # Recurse
                if col_idx < k:
                    if backtrack(row_idx, col_idx + 1):
                        return True
                else:
                    if backtrack(row_idx + 1, 1):
                        return True
                
                # Undo
                row[col_idx] = 0
                used_diffs ^= new_diffs
            
            return False
        
        # Start backtracking
        backtrack(0, 1)
        return best_rows is not None

    # Fallback greedy solution
    def greedy_solution():
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

    # First, get a greedy solution to establish an upper bound
    greedy = greedy_solution()
    best_rows = greedy
    best_max = max(max(row) for row in greedy)
    
    # Now try to improve with backtracking
    # Start with the greedy max as the limit, and try to find better solutions
    current_limit = best_max
    
    # Try backtracking with different seeds and limits
    seed = 0
    while time.time() < deadline:
        seed += 1
        
        # Try to find a solution with max < current_limit
        target_limit = current_limit - 1
        if target_limit < lower_bound:
            break
        
        # Save current best
        old_best_rows = best_rows
        old_best_max = best_max
        
        # Reset best to allow finding a new solution
        # Actually, we want to find ANY solution with max <= target_limit
        # Let's modify the approach: find a solution, and if it's better, keep it
        
        # We'll use a flag to track if we found a solution
        found = solve_with_backtracking(seed, target_limit)
        
        if found and best_max < old_best_max:
            current_limit = best_max
        elif not found:
            # No solution found with this limit, try a larger limit
            # But we already have a solution with old_best_max
            # So we can stop if we can't improve
            if target_limit == lower_bound:
                break
            # Try with a slightly larger limit? No, that doesn't make sense.
            # If we can't find a solution with max <= target_limit, we can't improve.
            # So we should stop.
            break
    
    # If backtracking didn't improve, return the greedy solution
    return best_rows
# EVOLVE-BLOCK-END
