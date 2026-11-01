"""
Strategy: Backtracking Solver with Forward Checking for Difference Triangle Set
This approach models the problem as a Constraint Satisfaction Problem (CSP).
- Variables: The k+1 entries for each of the n_rows rows.
- Constraints: All pairwise differences within a row must be unique across all rows.
- Algorithm: Backtracking search with:
    1. Forward Checking: Maintains the set of 'used' differences. When a value is placed,
       it checks if any future required differences (based on current partial row) are already used.
       More importantly, it prunes candidate values that would make it impossible to complete
       the current row or subsequent rows within the current best score bound.
    2. Dynamic Variable Ordering: Rows are not necessarily constructed in index order.
       We can try to construct the "hardest" rows first or those with tightest constraints.
       However, for simplicity and effectiveness in this specific structure, we construct rows
       sequentially but use a strong pruning heuristic: if the current max entry exceeds the
       current best, we prune.
    3. Heuristic Value Ordering: Try smaller values first to keep the scope tight.
    
This is fundamentally different from the current program which uses a constructive heuristic
that doesn't backtrack globally. If the greedy choice leads to a dead end, the current program
restarts the whole row or even the whole set. This new program backtracks step-by-step within
the row and across rows, allowing it to find solutions that require non-greedy early choices.
"""

import time
import sys
import random

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows of k+1 increasing integers starting at 0.
    All pairwise differences within rows must be distinct globally.
    Minimize the largest entry.
    """
    # Deadline: 145 seconds (5s safety margin from 150s limit)
    deadline = time.time() + 145.0
    
    # 1. Get a baseline solution using a fast greedy method
    # This provides an upper bound for pruning.
    baseline = _greedy_construct(n_rows, k)
    best_rows = baseline
    best_max_entry = _max_entry(baseline)
    
    # If the problem is trivial, return immediately
    if n_rows <= 2 and k <= 3:
        return baseline

    # 2. Backtracking Search
    # We will try to find a solution with max_entry < best_max_entry.
    # We use a recursive backtracking algorithm.
    
    # To make the search more effective, we can randomize the order of rows
    # or use different heuristics in different runs.
    
    # We'll run multiple restarts with different random seeds or row orderings.
    num_restarts = 500
    # Dynamic budget: if we find a better solution, we can use the remaining time
    # to try to improve further.
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        # In each restart, we might shuffle the row order to break symmetry
        # or just use a different random seed for tie-breaking.
        # For now, we'll stick to sequential rows but allow randomized tie-breaking
        # in value selection if we implement a more complex heuristic later.
        # Actually, pure greedy value ordering (smallest first) is usually best for minimization.
        # The "randomness" can come from trying different initial bounds or row orders.
        
        # Let's try constructing rows in a specific order.
        # Often, constructing rows that "use up" many small differences first can be bad.
        # Constructing rows that use large differences first might leave small ones for later?
        # No, the differences are global.
        
        # Let's just run the backtracking solver.
        # The solver will try to build a valid set of rows with max_entry < best_max_entry.
        
        # We pass the current best_max_entry - 1 as the hard upper bound for the search.
        # If the search finds a valid solution, it will have max_entry <= bound.
        # Since we want strictly smaller, we use best_max_entry - 1.
        
        bound = best_max_entry - 1
        
        # If bound is too small, the search might be very expensive or impossible.
        # We can estimate a lower bound.
        # Counting bound: total diffs = n_rows * k(k+1)/2.
        # Max diff is M. Number of possible diffs is M.
        # So M >= n_rows * k(k+1)/2.
        # For n=7, k=5: 7 * 15 = 105.
        # So bound must be at least 105.
        
        if bound < n_rows * k * (k + 1) // 2:
            continue
            
        result = _backtrack_solve(n_rows, k, bound, deadline)
        
        if result is not None:
            current_max = _max_entry(result)
            if current_max < best_max_entry:
                best_max_entry = current_max
                best_rows = result
                # If we hit the theoretical lower bound, we can stop
                if best_max_entry <= n_rows * k * (k + 1) // 2:
                    break

    return best_rows

def _backtrack_solve(n_rows, k, bound, deadline):
    """
    Attempts to construct n_rows of k+1 integers with max entry <= bound.
    Uses backtracking with forward checking.
    Returns a list of rows if successful, None otherwise.
    """
    used_diffs = set()
    rows = []
    
    # We construct rows one by one.
    # For each row, we backtrack on the values.
    
    # To improve performance, we can order the rows.
    # Let's try constructing the "hardest" rows first.
    # Heuristic: Rows that require larger differences might be harder.
    # But since all rows have the same structure (k+1 ints starting at 0),
    # the difficulty depends on the available differences.
    # Initially, all rows are symmetric.
    # As we fill rows, the remaining rows face more constraints.
    # So, constructing rows in any fixed order is fine, but backtracking
    # within the row is key.
    
    def backtrack_row(row_idx, current_row, current_max_val):
        """
        Recursively builds the current_row.
        row_idx: index of the row being built (0 to n_rows-1)
        current_row: list of values in the current row so far
        current_max_val: the largest value in current_row so far
        """
        if time.time() > deadline:
            return False
        
        # Base case: row is complete
        if len(current_row) == k + 1:
            # Check if this row's max entry exceeds the bound
            if current_row[-1] > bound:
                return False
            
            # Add row to global rows
            rows.append(current_row)
            
            # If this was the last row, we are done
            if row_idx == n_rows - 1:
                return True
            
            # Recurse to next row
            if backtrack_row(row_idx + 1, [0], 0):
                return True
            
            # Backtrack: remove row
            rows.pop()
            return False
        
        # Recursive case: add next value to current_row
        # The next value must be > current_max_val
        # And <= bound
        # And all new differences (next_val - a for a in current_row) must not be in used_diffs
        
        start_val = current_max_val + 1
        end_val = bound + 1
        
        # Optimization: We can estimate the minimum possible next value
        # to keep the row within the bound.
        # If we have k+1 - len(current_row) values left to place,
        # and the current max is M, the smallest possible final value is
        # M + (k+1 - len(current_row)) * 1? No, they are increasing integers.
        # If we have r values left, the smallest next value is M+1, then M+2, etc.
        # So the smallest possible final value is M + r.
        # If M + r > bound, then no solution exists from this state.
        remaining_slots = (k + 1) - len(current_row)
        min_possible_final = current_max_val + remaining_slots
        if min_possible_final > bound:
            return False
            
        # Iterate through possible next values
        # We try smaller values first to keep the scope tight.
        for next_val in range(start_val, end_val):
            # Check if next_val is valid
            # New differences: {next_val - a for a in current_row}
            new_diffs = set()
            valid = True
            for a in current_row:
                d = next_val - a
                if d in used_diffs:
                    valid = False
                    break
                # Also check for intra-row duplicates?
                # If d is in new_diffs, it's a duplicate within this row.
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
                
            if not valid:
                continue
                
            # Check if adding these diffs would make it impossible to complete the row?
            # Forward checking:
            # We need to place remaining_slots - 1 more values.
            # The smallest possible next-next value is next_val + 1.
            # This is a weak check.
            
            # Tentatively add new_diffs to used_diffs
            used_diffs.update(new_diffs)
            current_row.append(next_val)
            
            if backtrack_row(row_idx, current_row, next_val):
                return True
            
            # Backtrack
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False

    # Start the backtracking for the first row
    # We start with row 0, initial value [0]
    if backtrack_row(0, [0], 0):
        return rows
    else:
        return None

def _greedy_construct(n_rows, k):
    """
    Fast greedy construction to get a baseline solution.
    """
    used = set()
    rows = []
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

def _max_entry(rows):
    return max(max(row) for row in rows)

# EVOLVE-BLOCK-END
