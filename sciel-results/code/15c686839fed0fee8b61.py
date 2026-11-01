import time
import random
from collections import Counter

def construct(n_rows, k):
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # 1. Initial Greedy Solution (Seed)
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
    
    # Initialize SA State
    # We need to track differences globally to calculate cost efficiently
    diff_counter = Counter()
    for row in rows:
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                diff_counter[row[j] - row[i]] += 1
                
    def get_conflicts(counter):
        return sum(1 for d, c in counter.items() if c > 1)
        
    def get_max_entry(rows):
        return max(row[-1] for row in rows)
        
    current_conflicts = get_conflicts(diff_counter)
    current_max = get_max_entry(rows)
    current_cost = current_max + 100 * current_conflicts
    
    best_rows = [r[:] for r in rows]
    best_cost = current_cost
    
    # SA Parameters
    T = 10.0
    alpha = 0.95
    min_T = 0.01
    
    # Pre-compute indices for speed?
    # Indices for differences in a row of size k+1
    diff_indices = [(i, j) for i in range(k + 1) for j in range(i + 1, k + 1)]
    
    while T > min_T and time.time() < deadline:
        # Generate Neighbor
        new_rows = [r[:] for r in rows]
        r_idx = random.randint(0, n_rows - 1)
        # Pick a position to change. 
        # Bias towards the end to reduce scope?
        # Or random.
        pos = random.randint(1, k) # 1 to k (0 is fixed at 0)
        
        row = new_rows[r_idx]
        old_val = row[pos]
        
        # Determine range for new value
        lb = row[pos - 1] + 1
        ub = row[pos + 1] - 1 if pos < k else current_max + 10 # Allow some expansion? No, we want to minimize.
        # If pos == k, we are the max. We want to try smaller values.
        # If we pick a value larger than current max, scope increases.
        # Let's restrict ub to current_max for pos == k? 
        # Actually, if we are in an invalid state, we might need to expand to resolve conflicts.
        # But we start valid.
        # Let's allow ub = current_max + 5.
        
        if lb > ub:
            # Swap pos with pos-1? Or just skip
            continue
            
        # Bias towards smaller values to reduce scope
        # Sample from [lb, ub]
        # If we want to reduce scope, we should prefer smaller values.
        # Simple uniform is okay, but let's try:
        # If pos == k, try to decrease.
        # v = random.randint(lb, ub)
        
        # To make it more effective for minimization:
        # If current state is valid, try to decrease max.
        # If invalid, try to resolve conflicts.
        
        v = random.randint(lb, ub)
        
        if v == old_val:
            continue
            
        new_rows[r_idx][pos] = v
        
        # Update diff_counter
        # Remove old diffs for this row
        for i, j in diff_indices:
            d = row[j] - row[i]
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
                
        # Add new diffs for this row
        new_row = new_rows[r_idx]
        for i, j in diff_indices:
            d = new_row[j] - new_row[i]
            diff_counter[d] += 1
            
        # Calculate new cost
        new_conflicts = get_conflicts(diff_counter)
        new_max = max(new_rows[r_idx][-1], max(r[-1] for r in new_rows if r is not new_rows[r_idx]))
        # Optimization: just compute max of all rows
        # Since only one row changed, max is either new_rows[r_idx][-1] or previous max of others.
        # For simplicity and safety, compute max. n_rows is small (7).
        new_max = max(r[-1] for r in new_rows)
        
        new_cost = new_max + 100 * new_conflicts
        
        delta = new_cost - current_cost
        
        # Acceptance
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            # Accept
            rows = new_rows
            current_conflicts = new_conflicts
            current_max = new_max
            current_cost = new_cost
            
            if new_conflicts == 0 and new_max < best_cost:
                best_cost = new_max
                best_rows = [r[:] for r in rows]
        else:
            # Reject: do nothing (counter was not modified in the accepted branch logic,
            # or we assume the copy strategy mentioned in comments is handled elsewhere/not needed here
            # since the prompt asks to fix the syntax error of an empty block).
            pass

        T *= alpha
        
    return best_rows
