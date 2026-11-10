"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # 1. Generate initial valid solution using Greedy
    def greedy_row(used):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        return row, used

    used = set()
    rows = []
    for _ in range(n_rows):
        row, used = greedy_row(used)
        rows.append(row)
    
    current_scope = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    # Precompute valid diffs for speed (optional, but we check on the fly)
    # We need to check if a move creates a conflict.
    # A move changes one value in one row.
    # This affects diffs involving that value.
    
    # 2. Simulated Annealing
    # State: list of rows
    # Objective: Minimize max(max(r) for r in rows)
    # Move: Pick a random row, pick a random index (1..k), pick a new value.
    # Constraint: Row must remain strictly increasing and all global diffs distinct.
    
    # To make this efficient, we can restrict moves to "small" changes or
    # specifically try to reduce the scope.
    # However, pure random moves in a huge space are inefficient.
    # Strategy: 
    # - Maintain the set of used diffs.
    # - When proposing a move, we must remove old diffs and add new ones.
    # - If a collision occurs, reject.
    
    # Efficient diff management:
    # Since k is small (e.g., 5 for (7,5) task? No, task says (7,5) means 7 rows, 5 diffs? 
    # "Choose 7 rows of 6 increasing integers" -> k=5 in the prompt's construct(n_rows, k) 
    # where row has k+1 ints. So row length is 6.
    # Diffs per row: 6*5/2 = 15. Total diffs: 7*15 = 105.
    # Scope ~ 100-200.
    
    # We will implement a standard SA loop.
    # To speed up, we'll use a set for used_diffs.
    
    def get_used_diffs(rls):
        u = set()
        for r in rls:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    u.add(r[j] - r[i])
        return u

    current_used = get_used_diffs(rows)
    
    T = 10.0
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Pick a random row and position
        r_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)  # Indices 1..k (0 is fixed at 0)
        
        old_val = rows[r_idx][pos]
        
        # Propose new value
        # Bias towards reducing scope if old_val is the max
        # Or just random in a window
        # To ensure validity, we must check bounds and collisions.
        
        # Determine a range for new_val
        # Must be > rows[r_idx][pos-1] and < rows[r_idx][pos+1] (if exists)
        lower = rows[r_idx][pos-1] + 1
        upper = rows[r_idx][pos+1] - 1 if pos < k else current_scope + 10 # Allow expansion if needed? No, we want to minimize. 
        # Actually, if we want to minimize scope, we should try values <= current_scope.
        # But sometimes we need to move a large value to a smaller one, which might require moving others?
        # Single move SA is limited. Let's try to swap or shift.
        
        # Better move: Swap two values in the same row? No, breaks increasing order usually.
        # Move: Change one value.
        
        # If pos == k, it's the last element.
        # Let's try a value in [lower, min(upper, current_scope)]
        # If we find a valid move that reduces scope, accept.
        # If we find a valid move that increases scope, accept with prob exp(-delta/T).
        
        # Optimization: Only try to change the value if it's part of the "bottleneck" or random.
        
        # Generate candidate
        # Try a few candidates to increase acceptance rate
        accepted = False
        for _ in range(5):
            if upper < lower:
                break
            # Random value in range
            # Bias towards lower values to reduce scope
            # new_val = random.randint(lower, upper)
            # Let's try to target the current scope or below
            if upper > current_scope:
                upper = current_scope
            
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue
                
            # Check local validity (increasing order)
            if pos > 0 and new_val <= rows[r_idx][pos-1]:
                continue
            if pos < k and new_val >= rows[r_idx][pos+1]:
                continue
            
            # Check global diff validity
            # Remove old diffs involving (r_idx, pos)
            r = rows[r_idx]
            old_diffs_to_remove = set()
            for j in range(len(r)):
                if j != pos:
                    d = abs(r[j] - old_val)
                    old_diffs_to_remove.add(d)
                    
            # Add new diffs
            new_diffs_to_add = set()
            for j in range(len(r)):
                if j != pos:
                    d = abs(r[j] - new_val)
                    new_diffs_to_add.add(d)
            
            # Check collisions
            # Any new diff must not be in (current_used - old_diffs_to_remove)
            available = current_used - old_diffs_to_remove
            if new_diffs_to_add & available:
                continue
            
            # Valid move
            # Calculate new scope
            # Only the max of this row might change
            old_row_max = max(r)
            # New row max
            if new_val > old_row_max:
                new_row_max = new_val
            else:
                # If old_val was the max, we need to recompute max of row
                if old_val == old_row_max:
                    new_row_max = max(r[:pos] + [new_val] + r[pos+1:])
                else:
                    new_row_max = old_row_max
            
            # Global new scope
            # This is expensive to compute exactly for every move if we don't track maxes.
            # Approximation: 
            # If new_val > current_scope, new_scope = new_val
            # If old_val == current_scope and new_val < current_scope, new_scope = max(max(r_new), max(other_rows))
            # We can maintain a list of row maxes.
            
            # For simplicity and speed, let's just accept valid moves that don't increase the *potential* scope too much?
            # No, let's compute the exact new scope efficiently.
            
            # Maintain row_maxes
            # We don't have row_maxes array in scope. Let's compute it.
            # Since n_rows is small (7), max(r) is fast.
            
            # Actually, let's just do the swap and check validity of the whole set? No, too slow.
            
            # Let's use the diff set logic which is robust.
            
            # Apply move temporarily
            r[pos] = new_val
            
            # Update current_used
            current_used -= old_diffs_to_remove
            current_used |= new_diffs_to_add
            
            # Calculate new scope
            new_scope = max(max(rr) for rr in rows)
            
            # Acceptance
            delta = new_scope - current_scope
            if delta <= 0 or random.random() < pow(2.71828, -delta / T):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [rr[:] for rr in rows]
                accepted = True
                break
            else:
                # Revert
                r[pos] = old_val
                current_used -= new_diffs_to_add
                current_used |= old_diffs_to_remove
        
        if not accepted:
            # Try to cool down if stuck?
            T *= cooling
            
        # Small cooling
        T *= 0.9995
        
        if T < 0.01:
            # Restart with higher T? Or stop?
            # Let's restart with a fresh greedy if we haven't found the theoretical bound?
            # For now, just continue with low T (greedy descent)
            pass

    return best_rows
# EVOLVE-BLOCK-END
