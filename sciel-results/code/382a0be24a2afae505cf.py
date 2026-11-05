import time
import math
import sys
import random

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    time_limit = 115  # Leave margin
    
    # 1. Get an initial upper bound using a simple greedy
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                found = False
                # Try small values first to keep max low
                for x in range(row[-1] + 1, 200):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) != len(row):
                        continue
                    if new_diffs & used:
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    # Fallback for greedy: expand search
                    for x in range(row[-1] + 1, 500):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                if not found:
                    # Should not happen for small n,k, but just in case
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            row.append(x)
                            used.update(new_diffs)
                            break
                        x += 1
            rows.append(row)
        return rows

    initial_rows = greedy_init()
    initial_max = max(max(r) for r in initial_rows)
    
    best_rows = [list(r) for r in initial_rows]
    best_max = initial_max
    
    # 2. Backtracking search to improve
    # We will try to find solutions with max < current best_max
    
    def backtrack(row_idx, pos_in_row, current_row, row_diffs, global_diffs, target_max):
        """
        Returns True if a valid configuration is found, updating global_diffs and storing the solution.
        """
        if time.time() - start_time > time_limit:
            return False
            
        if row_idx == n_rows:
            # All rows constructed
            # Store solution
            # We need to pass the full solution up. 
            # Since Python passes by reference for lists, we can maintain a global 'solution' list.
            nonlocal solution_rows
            solution_rows = [list(r) for r in current_rows_list] # Need to manage this carefully
            return True

        if pos_in_row == k + 1:
            # Row is complete
            # Add row_diffs to global_diffs
            global_diffs.update(row_diffs)
            # Save current row
            current_rows_list.append(list(current_row))
            
            # Recurse to next row
            if backtrack(row_idx + 1, 0, [0], set(), global_diffs, target_max):
                return True
            
            # Backtrack
            current_rows_list.pop()
            for d in row_diffs:
                global_diffs.discard(d)
            return False
        
        # Determine next value x
        if pos_in_row == 0:
            # First element is always 0
            x = 0
            # No diffs added yet
            if backtrack(row_idx, pos_in_row + 1, current_row + [x], row_diffs, global_diffs, target_max):
                return True
            return False
        
        # pos_in_row > 0
        min_x = current_row[-1] + 1
        max_x = target_max
        
        # Heuristic: Try smaller values first to keep the row compact
        # This helps in finding a solution within target_max
        for x in range(min_x, max_x + 1):
            # Check if x is within target_max
            if x > target_max:
                break
                
            # Calculate new diffs
            new_diffs = set()
            valid = True
            for y in current_row:
                d = x - y
                if d in new_diffs:
                    valid = False
                    break
                if d in row_diffs:
                    valid = False
                    break
                if d in global_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Try x
            if backtrack(row_idx, pos_in_row + 1, current_row + [x], row_diffs | new_diffs, global_diffs, target_max):
                return True
                
        return False

    # Wrapper to manage state
    solution_rows = None
    current_rows_list = []
    
    # We want to find the minimum max.
    # Start from best_max - 1 downwards
    # But we must ensure we don't take too long.
    # We will run the backtracker for a limited time for each target_max?
    # Or just one big run that tries to minimize?
    # Better: Iterate target_max from best_max - 1 down to LB.
    # If a solution is found, update best and continue.
    # If time runs out, return best found.
    
    # Calculate a simple lower bound
    # Total diffs = n_rows * k * (k+1) / 2
    # Each diff is >= 1.
    # Max entry M allows diffs up to M.
    # Counting bound: roughly sqrt(n_rows * k^2 / 2)
    lb = int(math.sqrt(n_rows * k * (k+1) / 2)) + 1
    
    # We'll try to improve from best_max - 1 down to lb
    # To avoid getting stuck in one deep search, we can limit the time per target_max
    # Or just let it run until the global time limit.
    
    target_max = best_max - 1
    
    while target_max >= lb and time.time() - start_time < time_limit:
        # Reset state for this attempt
        solution_rows = None
        current_rows_list = []
        global_diffs_initial = set()
        
        # Run backtracking
        # We need to capture the solution if found
        found = backtrack(0, 0, [], set(), global_diffs_initial, target_max)
        
        if found:
            # Solution found with max <= target_max
            # The actual max might be less than target_max, but we know it's valid
            # Let's calculate the actual max of the found solution
            actual_max = max(max(r) for r in solution_rows)
            if actual_max < best_max:
                best_max = actual_max
                best_rows = solution_rows
                # We can try to go lower
                target_max = actual_max - 1
            else:
                # This shouldn't happen if we are decrementing, but just in case
                target_max -= 1
        else:
            # No solution found for this target_max
            # Try a smaller target_max? No, if no solution for T, no solution for T-1?
            # Wait, if no solution for max <= T, then definitely no solution for max <= T-1.
            # So we can break?
            # NO. The backtracker might have failed due to time limit or search order, not because it's impossible.
            # However, if it exhausted the search space, then it is impossible.
            # Our backtracker returns False if it exhausts the space OR if time runs out.
            # We need to distinguish.
            # For simplicity, if it returns False, we assume it's hard/impossible and try T-1?
            # Actually, if it's impossible for T, it's impossible for T-1.
            # But if it timed out, it might be possible.
            # Given the small size, if it doesn't find a solution quickly, it's likely not there.
            # Let's just decrement and try again, but with a fresh start.
            target_max -= 1
            
    return best_rows
