import sys
import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    # Greedy initialization
    def greedy(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Try to find the smallest next number
                # We need to ensure differences with existing row elements are not in 'used'
                # and differences within the new row are not in 'used'
                # Actually, greedy usually just adds the smallest x such that {x-a for a in row} is disjoint from 'used'
                # But we also need to ensure that adding x doesn't create duplicates within the row itself?
                # No, row is strictly increasing, so x-a > 0.
                # The condition is: for all a in row, (x-a) not in used.
                # And we add all (x-a) to used.
                # Wait, the "distinct over all rows" means the set of ALL differences from ALL rows must be unique.
                # So yes, 'used' is the global set of all differences found so far.
                
                x = current_max + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used.update(new_diffs)
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy(n_rows, k)
    
    # State for SA
    current_rows = [list(r) for r in best_rows]
    
    # Precompute differences for current state
    # diff_counts[d] = number of times difference d appears across all rows
    diff_counts = defaultdict(int)
    
    def update_diffs(row, index, old_val, new_val, is_remove):
        # Update diff_counts for changes in a specific row at a specific index
        # is_remove: if True, we are removing old_val and adding new_val? No, let's just handle add/remove explicitly.
        pass # We'll implement inline for speed

    # Initialize diff_counts
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                diff_counts[d] += 1
                
    def get_conflicts():
        # Count how many differences have count > 1
        # This is O(U) where U is number of unique diffs. U is approx N*K*(K+1)/2 which is small.
        return sum(1 for c in diff_counts.values() if c > 1)

    current_conflicts = get_conflicts()
    current_scope = max(max(row) for row in current_rows)
    
    # Cost function
    def get_cost(scope, conflicts):
        if conflicts == 0:
            return scope
        return 10000 + conflicts

    current_cost = get_cost(current_scope, current_conflicts)
    best_cost = current_cost
    best_rows_found = [list(r) for r in current_rows]
    
    start_time = time.time()
    time_limit = 115.0
    
    T = 50.0
    cooling_rate = 0.9999 # Slower cooling to explore more
    
    # Helper to apply a move and return deltas
    # We modify current_rows and diff_counts in place, then revert if rejected.
    
    while time.time() - start_time < time_limit:
        r = random.randint(0, n_rows - 1)
        i = random.randint(1, k)
        
        row = current_rows[r]
        old_val = row[i]
        
        # Determine range
        min_val = row[i-1] + 1
        max_val = row[i+1] - 1 if i < k else current_scope + 20
        
        if min_val > max_val:
            continue
            
        # Random perturbation
        # Bias towards small changes? Or uniform?
        # Let's try uniform in range, but if range is huge, maybe pick near old_val?
        # For now, uniform is standard.
        new_val = random.randint(min_val, max_val)
        
        if new_val == old_val:
            continue
            
        # Calculate differences to remove (involving old_val)
        # and differences to add (involving new_val)
        # Only differences within the same row are affected.
        
        # Remove old differences
        for j in range(len(row)):
            if j == i: continue
            if j < i:
                d = old_val - row[j]
                diff_counts[d] -= 1
            else:
                d = row[j] - old_val
                diff_counts[d] -= 1
                
        # Add new differences
        for j in range(len(row)):
            if j == i: continue
            if j < i:
                d = new_val - row[j]
                diff_counts[d] += 1
            else:
                d = row[j] - new_val
                # Note: row is increasing, so if j > i, row[j] > new_val is NOT guaranteed if new_val is large?
                # Wait, we enforced min_val = row[i-1]+1 and max_val = row[i+1]-1 (if i<k).
                # So row[i-1] < new_val < row[i+1].
                # Thus for j < i, row[j] < row[i-1] < new_val, so d = new_val - row[j] > 0.
                # For j > i, row[j] > row[i+1] > new_val, so d = row[j] - new_val > 0.
                # So the logic holds.
                diff_counts[d] += 1
                
        # Update row
        row[i] = new_val
        
        # Calculate new scope
        # Only row r changed.
        # New max of row r is max(row).
        # New global scope is max of all rows' maxes.
        # We can just calculate max(max(row) for row in current_rows)
        # Since n_rows is small (4), this is fast.
        new_scope = max(max(rw) for rw in current_rows)
        
        # Calculate new conflicts
        # We can update conflicts incrementally or recalculate.
        # Recalculating is O(U). U is small.
        # Let's try incremental for max speed.
        # However, tracking conflicts incrementally requires knowing which diffs went from 1->2 or 2->1.
        # Let's just recalculate for simplicity first, as U is ~ 4*36 = 144 max.
        new_conflicts = get_conflicts()
        
        new_cost = get_cost(new_scope, new_conflicts)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            # Accept
            current_scope = new_scope
            current_conflicts = new_conflicts
            current_cost = new_cost
            
            if new_conflicts == 0 and new_scope < best_cost:
                best_cost = new_scope
                best_rows_found = [list(rw) for rw in current_rows]
        else:
            # Revert
            # Remove new diffs
            for j in range(len(row)):
                if j == i: continue
                if j < i:
                    d = new_val - row[j]
                    diff_counts[d] -= 1
                else:
                    d = row[j] - new_val
                    diff_counts[d] -= 1
                    
            # Add old diffs
            for j in range(len(row)):
                if j == i: continue
                if j < i:
                    d = old_val - row[j]
                    diff_counts[d] += 1
                else:
                    d = row[j] - old_val
                    diff_counts[d] += 1
                    
            row[i] = old_val
            
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return best_rows_found
