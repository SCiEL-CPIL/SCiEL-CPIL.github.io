import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    # Helper to calculate differences for a row
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    # 1. Initialize with Greedy Solution (Valid, D=0)
    # This ensures we start in a valid state.
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
    
    # If n_rows is 0 or k is 0, handle edge cases
    if n_rows == 0 or k == 0:
        return [[0] * (k+1) for _ in range(n_rows)]

    # Calculate initial cost
    # We need to track diff counts globally
    diff_count = {}
    for r in rows:
        for d in get_diffs(r):
            diff_count[d] = diff_count.get(d, 0) + 1
    
    def calc_D(diff_count):
        d_sum = 0
        for count in diff_count.values():
            if count > 1:
                d_sum += count - 1
        return d_sum
    
    D = calc_D(diff_count)
    M = max(max(r) for r in rows)
    
    # Cost function: M + lambda * D
    LAMBDA = 1000
    
    def get_cost(M, D):
        return M + LAMBDA * D
    
    best_rows = [r[:] for r in rows]
    best_cost = get_cost(M, D)
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    alpha = 0.999 # Cooling rate
    
    current_rows = [r[:] for r in rows]
    current_M = M
    current_D = D
    
    # Precompute max and second max for each row to update M efficiently?
    # Or just recalculate max for the modified row. O(k) is fine.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick random row and position
        i = random.randint(0, n_rows - 1)
        j = random.randint(1, k) # Position 1..k (0 is fixed at 0)
        
        row = current_rows[i]
        old_val = row[j]
        
        # Determine range for new value
        # Must be strictly increasing
        lower = row[j-1] + 1
        # Upper bound: let's try to keep it near current M or slightly larger
        # If j < k, upper is row[j+1] - 1
        # If j == k, upper can be larger, e.g., current_M + 10
        if j < k:
            upper = row[j+1] - 1
        else:
            upper = max(current_M + 10, lower + 10)
            
        if lower >= upper:
            T *= alpha
            continue
            
        new_val = random.randint(lower, upper)
        if new_val == old_val:
            T *= alpha
            continue
            
        # Calculate change in D
        # Remove old diffs involving j
        old_diffs = []
        for m in range(k+1):
            if m != j:
                d = abs(row[j] - row[m])
                old_diffs.append(d)
                
        # Count how many of these are duplicates currently
        # We need to update diff_count
        # To avoid modifying diff_count if we reject, we calculate delta
        
        # Number of duplicates removed:
        # For each d in old_diffs, if diff_count[d] > 1, removing one reduces D by 1.
        # If diff_count[d] == 1, removing one makes it 0, D unchanged.
        
        # Let's simulate the counts
        # Actually, it's easier to just apply, calculate, and revert if rejected?
        # Given k is small, applying and reverting is O(k).
        
        # Apply move
        row[j] = new_val
        
        # Update diff_count
        for d in old_diffs:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d] # Optional cleanup
        
        new_diffs = []
        for m in range(k+1):
            if m != j:
                d = abs(new_val - row[m])
                new_diffs.append(d)
                diff_count[d] = diff_count.get(d, 0) + 1
        
        # Calculate new M
        new_row_max = max(row)
        # Global M:
        # If old_val was the global max, we need to check if new_val is still max or if another row has max.
        # To be safe and simple, recalculate global M?
        # M = max(max(r) for r in current_rows)
        # This is O(n_rows * k). n_rows=7, k=6. Very fast.
        new_M = max(max(r) for r in current_rows)
        
        new_D = calc_D(diff_count)
        
        new_cost = get_cost(new_M, new_D)
        old_cost = get_cost(current_M, current_D)
        
        delta_cost = new_cost - old_cost
        
        # Acceptance
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_M = new_M
            current_D = new_D
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [r[:] for r in current_rows]
        else:
            # Revert
            row[j] = old_val
            
            # Revert diff_count
            for d in new_diffs:
                diff_count[d] -= 1
                if diff_count[d] == 0:
                    del diff_count[d]
            
            for d in old_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
                
            # M and D revert implicitly as we didn't update current_M/D
            pass
            
        T *= alpha

    return best_rows
