"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import sys
import time
import itertools

def construct(n_rows, k):
    # Difference triangle set: choose n_rows rows of k+1 increasing integers starting at 0
    # such that all positive differences a_j - a_i (i < j) are distinct across all rows.
    # Minimize the largest entry.
    
    # For small cases, use a backtracking CSP solver with forward checking.
    # This is much more effective than greedy for finding the true minimum or near-minimum.
    
    start_time = time.time()
    time_limit = 100  # Leave 20s buffer for evaluation overhead
    
    # Initial upper bound from a simple greedy to have a starting point for pruning
    def greedy_construct(n_rows, k):
        rows = []
        used_diffs = set()
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        return max(max(row) for row in rows)
    
    best_max = greedy_construct(n_rows, k)
    best_rows = None
    
    # We will try to find a solution with max entry < best_max
    # The search space is defined by the rows.
    # We use a recursive backtracking approach.
    
    # To speed up, we can fix the first row to be as "tight" as possible or just let the search decide.
    # Actually, for DTS, the structure is symmetric. 
    # We'll search for rows one by one.
    
    # Optimization: Pre-compute valid differences? No, dynamic set is better.
    
    # We'll use a global variable for the best solution found
    global_best = {'rows': None, 'max_val': best_max}
    
    # Since the problem is small (n_rows=4, k=7 => 4 rows of 8 ints),
    # the search space is manageable with good pruning.
    
    # Let's try to improve upon the greedy solution.
    # We can use a randomized restart approach or just a deterministic deep search.
    # Deterministic is safer for reproducibility, but randomized might escape local minima if we were doing local search.
    # Here we are doing exact search (branch and bound).
    
    # We'll iterate with a decreasing limit to find the minimum.
    # Start from best_max - 1 down to a lower bound.
    # Lower bound: simple counting bound is 112 for (4,7). 
    # But we don't know the exact lower bound for other (n,k).
    # Let's just do a Branch and Bound search minimizing the max element.
    
    # State: 
    # rows: list of completed rows
    # current_row: list of elements in current row
    # used_diffs: set of all differences used so far
    
    # To make it faster, we can order the candidates for the next element.
    # The next element must be > last element.
    # And its differences with all previous elements in the row must not be in used_diffs.
    
    def backtrack(rows, current_row, used_diffs, depth):
        nonlocal global_best
        
        if time.time() - start_time > time_limit:
            return False
        
        if len(rows) == n_rows:
            # All rows constructed
            if current_row:
                max_val = max(max(r) for r in rows + [current_row])
            else:
                max_val = max(max(r) for r in rows)
            
            if max_val < global_best['max_val']:
                global_best['max_val'] = max_val
                global_best['rows'] = [list(r) for r in rows] + ([list(current_row)] if current_row else [])
            return True
        
        # If we just finished a row, start a new one
        if len(current_row) == k + 1:
            rows.append(list(current_row))
            if backtrack(rows, [0], used_diffs, depth + 1):
                rows.pop()
                return True
            rows.pop()
            return False
        
        # Otherwise, we are in the middle of a row
        last_val = current_row[-1]
        
        # Try next values
        # We can bound the next value.
        # The maximum value in the final solution must be < global_best['max_val']
        # So any element in the current row must be < global_best['max_val']
        
        start_x = last_val + 1
        end_x = global_best['max_val']
        
        for x in range(start_x, end_x + 1):
            # Check differences
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            
            # Forward checking: 
            # If we add x, do we block any future possibilities?
            # Simple check: just proceed.
            
            new_used_diffs = used_diffs | new_diffs
            if backtrack(rows, current_row + [x], new_used_diffs, depth):
                return True
        
        return False

    # The above backtracking might be slow if it explores too many branches.
    # For (4,7), the greedy max is likely around 134-140.
    # The optimal is 134.
    # The search space is small enough that this should work within 100s.
    
    # However, the `backtrack` function above has a subtle issue: 
    # It treats `current_row` as a list that grows. 
    # When `len(current_row) == k+1`, it moves to the next row.
    # This is correct.
    
    # Let's run it.
    backtrack([], [0], set(), 0)
    
    if global_best['rows']:
        return global_best['rows']
    else:
        # Fallback to greedy if search failed or timed out without finding anything better
        # (Though it starts with greedy best, so it always has a valid solution)
        # Re-run greedy to get the rows back
        rows = []
        used_diffs = set()
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
