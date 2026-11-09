import time
import random
import math

def construct(n_rows, k):
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # Leave 10s buffer for safety
    
    # 1. Greedy Initialization
    def greedy_construct(n_rows, k):
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
        return rows, used

    rows, used = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    
    # Calculate initial cost
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def count_collisions(rows):
        # This is expensive, so we maintain 'used' dynamically in the loop
        # For initial check or validation
        all_diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    all_diffs.append(r[j] - r[i])
        return len(all_diffs) - len(set(all_diffs))
    
    initial_scope = get_scope(rows)
    # We assume greedy is valid, so collisions = 0
    
    # Simulated Annealing
    T = 1000.0
    alpha = 0.9995
    min_T = 1e-6
    
    # Helper to calculate difference set for a row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Main SA Loop
    while time.time() - start_time < timeout:
        if T < min_T:
            break
            
        # Pick a random row and index
        r_idx = random.randint(0, n_rows - 1)
        # Index 0 is always 0, so pick from 1 to k
        pos = random.randint(1, k)
        
        row = rows[r_idx]
        old_val = row[pos]
        
        # Propose a move
        # Delta magnitude depends on temperature or is fixed small
        delta = random.randint(-10, 10)
        new_val = old_val + delta
        
        # Check validity of the row structure (strictly increasing)
        if pos > 0 and new_val <= row[pos-1]:
            continue
        if pos < k and new_val >= row[pos+1]:
            continue
        if new_val <= 0:
            continue
            
        # Calculate old diffs involving this position
        # Diffs with previous elements: old_val - row[i] for i < pos
        # Diffs with next elements: row[j] - old_val for j > pos
        old_diffs_to_remove = set()
        for i in range(pos):
            old_diffs_to_remove.add(old_val - row[i])
        for j in range(pos+1, len(row)):
            old_diffs_to_remove.add(row[j] - old_val)
            
        # Calculate new diffs
        new_diffs_to_add = set()
        for i in range(pos):
            new_diffs_to_add.add(new_val - row[i])
        for j in range(pos+1, len(row)):
            new_diffs_to_add.add(row[j] - new_val)
            
        # Check for internal collision in new_diffs_to_add
        # Since row is increasing, new_val - row[i] are distinct for different i.
        # row[j] - new_val are distinct for different j.
        # But could a diff from left collide with a diff from right?
        # new_val - row[i] == row[j] - new_val => 2*new_val == row[i] + row[j]
        # This is possible. So we must check if len(new_diffs_to_add) == number of pairs.
        num_left = pos
        num_right = len(row) - pos - 1
        if len(new_diffs_to_add) != num_left + num_right:
            continue # Internal collision in the row itself (or duplicate diff)
            
        # Check collision with global 'used' set
        # We need to check if any diff in new_diffs_to_add is in 'used' 
        # EXCEPT for the ones we are removing.
        
        # Optimization: 
        # 1. Remove old_diffs_to_remove from 'used'
        # 2. Check if new_diffs_to_add intersects with 'used'
        # 3. If yes, revert and reject (or accept with penalty if we allow invalid states, 
        #    but for simplicity and speed, let's stick to valid states first, 
        #    or use a penalty approach).
        
        # Given the density, valid-state SA might get stuck. 
        # Let's use a penalty approach.
        
        # Remove old
        for d in old_diffs_to_remove:
            used.discard(d)
            
        # Check collisions
        collisions = 0
        for d in new_diffs_to_add:
            if d in used:
                collisions += 1
            # Also check duplicates within new_diffs_to_add? 
            # We already checked len, so no internal dupes.
            
        # Add new
        for d in new_diffs_to_add:
            used.add(d)
            
        # Calculate Cost
        # Scope change
        old_scope = max(row) # Wait, scope is global max. 
        # We need global scope.
        # Let's precompute row maxes.
        
        # Actually, maintaining global scope efficiently is tricky if we don't track it.
        # Let's just compute global scope every time? 
        # n_rows=7, k=5. Max 42 elements. Max computation is fast.
        
        new_row = row[:]
        new_row[pos] = new_val
        rows[r_idx] = new_row
        
        current_scope = max(max(r) for r in rows)
        
        # Cost = Scope + Penalty * Collisions
        # We want to minimize Scope. Collisions should be 0.
        # If collisions > 0, we are in an invalid state.
        # Let's define Cost = current_scope + 1000 * collisions
        
        # We need the previous cost to compare.
        # We didn't store previous global scope or collisions.
        # This is inefficient.
        
        # Let's refactor to store current_cost components.
        
        # For now, let's just accept if valid (collisions==0) and scope is better, 
        # or accept invalid with probability.
        
        # To make it work, let's track:
        # global_scope
        # num_collisions (hard to track incrementally without a full recount or careful set logic)
        
        # Let's simplify: 
        # Only accept moves that keep the state VALID (collisions == 0).
        # This is "Valid State SA".
        
        if collisions == 0:
            # Valid move
            # Check if scope improved
            # We need to know if this move reduced the global max.
            # If new_val < old_val and old_val was the max of that row and that row was the global max, it might reduce scope.
            # If new_val > old_val, it might increase scope.
            
            # Let's just calculate the new global scope.
            new_global_scope = max(max(r) for r in rows)
            
            # We need the old global scope. Let's store it.
            # I will rewrite the loop to track state properly.
            
            pass # Placeholder
            
        else:
            # Revert
            rows[r_idx] = row # Restore original row list content? No, row was modified?
            # I did `new_row = row[:]`, so `row` is intact.
            # But I added to `used`. Need to remove `new_diffs_to_add` and restore `old_diffs_to_remove`.
            for d in new_diffs_to_add:
                used.discard(d)
            for d in old_diffs_to_remove:
                used.add(d)
            # Reject invalid move (for now)
            continue
            
        T *= alpha

    # Final Validation
    # Ensure the returned rows are valid
    all_diffs = set()
    valid = True
    for r in best_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    valid = False
                all_diffs.add(d)
                
    if not valid:
        # Fallback to greedy if SA broke it (shouldn't happen if we only accept valid moves)
        rows, _ = greedy_construct(n_rows, k)
        return rows
        
    return best_rows

# The above draft has issues with tracking global state (scope, collisions) efficiently.
# Let's write a cleaner version.
