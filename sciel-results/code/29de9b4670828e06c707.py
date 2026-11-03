import time
import random
import sys

def construct(n_rows, k):
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for the evaluator's 150s limit
    
    # 1. Initial Greedy Construction
    # This provides a valid starting point.
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # Helper to calculate scope
    def get_scope(rows):
        return max(max(r) for r in rows)
        
    # Helper to check if a set of differences is valid given a used set
    # and returns the new used set if valid
    def try_add_diffs(diffs, used_set):
        if diffs & used_set:
            return None
        return used_set | diffs

    # Simulated Annealing Loop
    # Temperature schedule
    T = 50.0
    cooling = 0.995
    
    # We need a fast way to check validity.
    # We maintain `used` set globally.
    # When proposing a move, we calculate the "delta" differences.
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Move: Try to change a single element in the row
        # We pick an index j > 0 to change
        # To keep it simple and valid, let's try to "shift" the last element first, 
        # or a random element.
        # Changing an internal element requires checking bounds with neighbors.
        
        # Let's try a move that picks a random index j (1..k) and tries a new value x_new
        # x_new must be > row[j-1] and < row[j+1] (if j < k)
        
        j = random.randint(1, k)
        
        # Determine range for x_new
        low = row[j-1] + 1
        high = row[j+1] - 1 if j < k else best_scope + 20  # Allow expansion if needed
        
        if low > high:
            continue
            
        # We want to try values near the current one or smaller to reduce scope
        # Current value
        x_old = row[j]
        
        # Propose new value
        # Bias towards smaller values if we are trying to minimize scope
        # But we must maintain strict increase.
        
        # Let's sample a few candidates
        found_valid = False
        # Try to find a valid x_new
        # We need to remove old diffs, check new diffs, and update used.
        
        # Old diffs involving x_old:
        # D_old = {x_old - a for a in row[:j]} U {b - x_old for b in row[j+1:]}
        
        # To speed up, we can just iterate through a small range of candidates
        # around x_old, biased towards lower values.
        
        # Candidate generation:
        # 1. Try x_old - delta
        # 2. Try x_old + delta
        # 3. Random in [low, high]
        
        # Let's try a few random steps
        for _ in range(5):
            if time.time() > deadline:
                break
                
            # Random candidate
            if random.random() < 0.5:
                # Try to shrink
                # Pick a value between low and x_old
                if x_old > low:
                    x_new = random.randint(low, x_old - 1)
                else:
                    x_new = x_old
            else:
                # Try to expand or random
                if high > x_old:
                    x_new = random.randint(x_old + 1, high)
                else:
                    x_new = x_old
                    
            if x_new == x_old:
                continue
                
            # Check bounds
            if x_new < low or x_new > high:
                continue
                
            # Calculate new differences
            # D_new = {x_new - a for a in row[:j]} U {b - x_new for b in row[j+1:]}
            
            # Optimization: 
            # Instead of creating sets every time, we can check intersection with used.
            # But we also need to ensure D_new doesn't intersect with itself? 
            # No, differences within a row are already distinct if the row is strictly increasing?
            # Wait. If row is strictly increasing, differences x_j - x_i are distinct?
            # Not necessarily. E.g., 0, 1, 3. Diffs: 1, 3, 2. Distinct.
            # 0, 2, 4. Diffs: 2, 4, 2. NOT distinct.
            # So we must ensure the new row has distinct internal differences.
            
            # Let's construct the new row temporarily
            new_row = row[:]
            new_row[j] = x_new
            
            # Check if new_row has distinct differences
            # And if these differences are disjoint from `used` (excluding old diffs)
            
            # Efficient check:
            # 1. Remove old diffs from `used`? No, don't modify global state yet.
            # 2. Check if new diffs intersect `used` - old_diffs.
            
            # Let's compute old_diffs and new_diffs
            # This is O(k^2) per move, which might be slow if k is large.
            # But k is usually small (e.g., 5 or 6 in the prompt example).
            
            old_diffs = set()
            for i in range(len(row)):
                for m in range(i+1, len(row)):
                    old_diffs.add(row[m] - row[i])
                    
            new_diffs = set()
            for i in range(len(new_row)):
                for m in range(i+1, len(new_row)):
                    d = new_row[m] - new_row[i]
                    # Check distinctness within row
                    if d in new_diffs:
                        break
                    new_diffs.add(d)
                else:
                    continue
                break # Inner break, need to handle logic properly
            
            # Proper distinctness check for new_row
            valid_internal = True
            new_diffs = set()
            for i in range(len(new_row)):
                for m in range(i+1, len(new_row)):
                    d = new_row[m] - new_row[i]
                    if d in new_diffs:
                        valid_internal = False
                        break
                    new_diffs.add(d)
                if not valid_internal:
                    break
            
            if not valid_internal:
                continue
                
            # Check against global used
            # The set of diffs currently in `used` includes the old_diffs of this row.
            # We need to check if (new_diffs - old_diffs) intersects (used - old_diffs)
            # Or simply: (new_diffs | (used - old_diffs)) == (used - old_diffs) | new_diffs?
            # Easier: 
            # 1. Remove old_diffs from a copy of used? No, too slow.
            # 2. Check if any d in new_diffs is in used, AND d is NOT in old_diffs.
            
            conflict = False
            for d in new_diffs:
                if d in used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Move is valid!
            # Calculate new scope
            new_scope = max(max(r) for r in rows) # Wait, rows[r_idx] is not updated yet
            # Update rows
            rows[r_idx] = new_row
            current_scope = max(max(r) for r in rows)
            
            # Update used
            used -= old_diffs
            used |= new_diffs
            
            # Acceptance
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
                
            # Metropolis criterion
            # Cost is scope. Lower is better.
            # Delta cost = current_scope - best_scope (if we compare to best)
            # Or compare to previous state?
            # Standard SA compares to previous state.
            # Let's keep track of prev_scope
            # Actually, simpler: accept if better. If worse, accept with prob exp(-delta/T)
            
            # We need prev_scope. Let's assume we always compare to the current state of `rows` before the move.
            # We didn't store prev_scope explicitly in the loop above, but we can infer.
            # Let's restructure slightly to track prev_state.
            
            # For now, let's just accept if it's better or equal. 
            # To escape local minima, we need to accept worse moves.
            # Let's restart the loop logic to be cleaner.
            
            break # Accept this move and break candidate loop
            
        # If we broke due to time, exit
        if time.time() > deadline:
            break
            
        # Cooling
        T *= cooling
        
    # Return the best valid solution found
    return best_rows
