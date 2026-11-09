import sys
import time
import random
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    # Start time
    start_time = time.time()
    timeout = 115  # seconds, leave buffer for overhead
    
    # 1. Generate initial greedy solution
    def greedy_init(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if x is valid
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, used

    current_rows, current_used = greedy_init(n_rows, k)
    
    # Calculate initial max
    def get_max(rows):
        return max(max(r) for r in rows)
    
    best_max = get_max(current_rows)
    best_rows = [r[:] for r in current_rows]
    
    # 2. Simulated Annealing
    # State: current_rows, current_used
    # Objective: Minimize max element
    
    # Helper to check if a change is valid
    def check_change(r_idx, j_idx, v_new):
        row = current_rows[r_idx]
        v_old = row[j_idx]
        
        # 1. Check bounds and sorted order
        if j_idx > 0 and v_new <= row[j_idx - 1]:
            return False, set(), set()
        if j_idx < len(row) - 1 and v_new >= row[j_idx + 1]:
            return False, set(), set()
        
        # 2. Check uniqueness in row
        if v_new in row and v_new != v_old:
            return False, set(), set()
            
        # 3. Compute differences to remove
        diffs_remove = set()
        for i in range(len(row)):
            if i == j_idx: continue
            d = abs(row[i] - v_old)
            diffs_remove.add(d)
            
        # 4. Compute differences to add
        diffs_add = set()
        for i in range(len(row)):
            if i == j_idx: continue
            d = abs(row[i] - v_new)
            diffs_add.add(d)
            
        # 5. Check for conflicts
        # The new diffs must not be in (current_used - diffs_remove)
        # And they must be unique among themselves (they are, because row elements are distinct)
        # But we must ensure they don't conflict with each other? 
        # d = |row[i] - v_new|. Since row[i] are distinct, these diffs are distinct.
        
        # The set of "available" diffs is current_used - diffs_remove
        # We check if diffs_add intersects with available
        if diffs_add & (current_used - diffs_remove):
            return False, diffs_remove, diffs_add
            
        return True, diffs_remove, diffs_add

    def apply_change(r_idx, j_idx, v_new, diffs_remove, diffs_add):
        row = current_rows[r_idx]
        v_old = row[j_idx]
        row[j_idx] = v_new
        current_used.difference_update(diffs_remove)
        current_used.update(diffs_add)
        
    def revert_change(r_idx, j_idx, v_old, diffs_remove, diffs_add):
        row = current_rows[r_idx]
        v_new = row[j_idx]
        row[j_idx] = v_old
        current_used.difference_update(diffs_add)
        current_used.update(diffs_remove)

    T = 50.0
    cooling = 0.99
    min_T = 0.01
    
    while time.time() - start_time < timeout:
        if T < min_T:
            break
            
        # Select a move
        # Bias towards rows with larger max
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        
        # Pick a random index to change (1 to k, index 0 is fixed at 0)
        # Bias towards the last element to reduce max
        # 70% chance to pick last element, 30% random
        if random.random() < 0.7:
            j_idx = k
        else:
            j_idx = random.randint(1, k)
            
        v_old = current_rows[r_idx][j_idx]
        
        # Determine range for v_new
        # Lower bound: row[j_idx-1] + 1
        # Upper bound: row[j_idx+1] - 1
        lb = current_rows[r_idx][j_idx - 1] + 1
        ub = current_rows[r_idx][j_idx + 1] - 1 if j_idx < k else v_old + 10 # Allow some expansion
        
        if lb > ub:
            continue
            
        # Try to find a new value
        # Strategy: 
        # 1. Try to shrink (v_new < v_old)
        # 2. Try to move randomly
        
        v_new = v_old
        valid = False
        diffs_remove = set()
        diffs_add = set()
        
        # Try a few random candidates
        # If we want to shrink, pick from [lb, v_old - 1]
        # If we want to perturb, pick from [lb, ub]
        
        # 50% chance to try shrinking, 50% random
        if random.random() < 0.5 and v_old > lb:
            # Try shrinking
            # Pick a random value in [lb, v_old - 1]
            # Or try the smallest valid one? Random is better for SA.
            # But checking one random value might fail often.
            # Let's try up to 3 random candidates.
            for _ in range(3):
                v_new = random.randint(lb, v_old - 1)
                valid, diffs_remove, diffs_add = check_change(r_idx, j_idx, v_new)
                if valid:
                    break
        else:
            # Random move
            for _ in range(3):
                v_new = random.randint(lb, ub)
                valid, diffs_remove, diffs_add = check_change(r_idx, j_idx, v_new)
                if valid:
                    break
        
        if not valid:
            continue
            
        # Calculate Delta E
        old_max = get_max(current_rows)
        # Simulate change to get new max? 
        # We can just check if v_new affects the max.
        # If v_old was the unique max and v_new < v_old, new_max < old_max.
        # If v_old was not the max, new_max = old_max (unless v_new > old_max, but v_new <= ub which is constrained by neighbors, usually < old_max if shrinking).
        
        # For simplicity and correctness, let's just apply, check, and revert if needed.
        # But calculating get_max is O(N*K). With N=4, K=7, it's very fast.
        
        apply_change(r_idx, j_idx, v_new, diffs_remove, diffs_add)
        
        new_max = get_max(current_rows)
        delta_E = new_max - old_max
        
        # Acceptance
        if delta_E < 0:
            accept = True
        else:
            # Accept with probability exp(-delta_E / T)
            # If delta_E is large, prob is low.
            if T > 0:
                prob = math.exp(-delta_E / T)
            else:
                prob = 0
            accept = random.random() < prob
            
        if accept:
            if new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in current_rows]
        else:
            # Revert
            revert_change(r_idx, j_idx, v_old, diffs_remove, diffs_add)
            
        T *= cooling

    return best_rows
