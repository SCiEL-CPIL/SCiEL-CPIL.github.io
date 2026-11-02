"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    timeout = 115  # Leave 5s margin
    
    best_solution = None
    best_max = float('inf')
    
    # Run multiple independent SA runs
    for run_id in range(50):
        if time.time() - start_time > timeout:
            break
            
        # Initial solution: Greedy
        sol = _greedy_initial(n_rows, k)
        if sol is None:
            continue
            
        # Simulated Annealing
        sol, final_max = _simulated_annealing(sol, n_rows, k, run_id, time.time(), timeout)
        
        if final_max < best_max:
            best_max = final_max
            best_solution = sol
            
        if best_max <= 134: # Best known bound
            break

    if best_solution is None:
        best_solution = _greedy_initial(n_rows, k)
        
    return best_solution

def _greedy_initial(n_rows, k):
    """Generate an initial valid solution using greedy."""
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

def _simulated_annealing(initial_sol, n_rows, k, seed, start_time, timeout):
    """
    Perform Simulated Annealing to minimize the max element.
    """
    rng = random.Random(seed * 999 + 123)
    rows = [list(r) for r in initial_sol]
    
    # Calculate initial global used diffs and max
    global_used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                global_used.add(row[j] - row[i])
                
    current_max = max(max(r) for r in rows)
    current_cost = current_max
    
    # Temperature schedule
    T = 10.0
    cooling_rate = 0.95
    min_T = 0.01
    
    # We will iterate until time limit or T is very low
    while T > min_T and (time.time() - start_time) < timeout:
        T *= cooling_rate
        
        # Perform a move
        # Pick random row and element
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Pick an element to move (index 1 to k, index 0 is fixed at 0)
        # We can also try to move the last element more aggressively to reduce max
        if rng.random() < 0.5:
            j = k # Focus on the largest element
        else:
            j = rng.randint(1, k)
            
        old_val = row[j]
        
        # Determine range for new value
        # Must be > row[j-1] and < row[j+1] (if j < k)
        low = row[j-1] + 1
        high = row[j+1] - 1 if j < k else old_val + 10 # Allow some expansion if needed, but we want to minimize
        
        # If we are trying to reduce max, and j==k, we should bias towards lower values
        # If j < k, we can move in either direction
        
        # Try a random new value in a local neighborhood
        # Neighborhood size depends on T? Or fixed?
        # Let's try a random offset
        offset_range = max(5, int(T))
        delta = rng.randint(-offset_range, offset_range)
        new_val = old_val + delta
        
        # Clamp to valid range
        if new_val < low:
            new_val = low
        if new_val > high:
            new_val = high
            
        if new_val == old_val:
            continue
            
        # Check validity of new_val
        # 1. Order is maintained by clamping
        # 2. Differences check
        # Old diffs involving j:
        old_diffs = set()
        for i in range(j):
            old_diffs.add(old_val - row[i])
        for i in range(j + 1, len(row)):
            old_diffs.add(row[i] - old_val)
            
        # New diffs involving j:
        new_diffs = set()
        for i in range(j):
            new_diffs.add(new_val - row[i])
        for i in range(j + 1, len(row)):
            new_diffs.add(row[i] - new_val)
            
        # Check if new_diffs intersect with global_used minus old_diffs
        # Note: global_used contains all diffs from all rows.
        # We need to ensure new_diffs are not in (global_used - old_diffs)
        # Also new_diffs must not contain 0 (impossible since strict inequality)
        
        # Optimization: 
        # available = global_used - old_diffs
        # if new_diffs & available: invalid
        
        # To avoid creating a new set 'available' every time, we can just check:
        # for d in new_diffs:
        #     if d in global_used and d not in old_diffs:
        #         invalid = True
        
        valid = True
        for d in new_diffs:
            if d in global_used and d not in old_diffs:
                valid = False
                break
                
        if not valid:
            continue
            
        # Calculate new cost
        # Only the max might change
        # If j == k (last element), new_max = max(current_max_without_last, new_val)
        # If j < k, new_max = current_max (unless old_val was the unique max and new_val < old_val? 
        #    Actually if j<k, row[j+1] > new_val, so max of row doesn't change unless j was the max? 
        #    No, j<k means there are elements after it. The max of the row is row[k].
        #    So if j < k, the max of the row is unchanged.
        #    If j == k, the max of the row becomes new_val.
        
        if j == k:
            new_row_max = new_val
            # Check if this row was the one contributing to global max
            # We just recalculate global max for simplicity, or track it.
            # Recalculating is O(n_rows * k). Acceptable.
            new_cost = max(max(r if r is not row else row[:k] + [new_val]) for r in rows)
            # Wait, `row` is a reference. We haven't updated it yet.
            # Let's just calculate max of all rows with the hypothetical change.
            # Faster:
            other_maxes = [max(r) for i, r in enumerate(rows) if i != r_idx]
            if other_maxes:
                other_max = max(other_maxes)
            else:
                other_max = 0
            new_cost = max(other_max, new_val)
        else:
            new_cost = current_cost
            
        # Acceptance
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
            # Accept move
            # Update global_used
            global_used.difference_update(old_diffs)
            global_used.update(new_diffs)
            
            # Update row
            row[j] = new_val
            
            current_cost = new_cost
            
    return rows, current_cost

# EVOLVE-BLOCK-END
