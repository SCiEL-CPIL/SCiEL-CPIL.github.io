"""Difference triangle set (n_rows, k) — Simulated Annealing with fast difference tracking."""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _scope_of(rows):
    """Return the largest entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _validate(rows, n_rows, k):
    """Check that rows form a valid difference set."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _greedy_init(n_rows, k, rng):
    """
    Fast greedy initialization.
    Builds rows one by one, picking the smallest valid integer for each position.
    This provides a valid starting point for the optimizer.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            # Try to find a valid x
            # Optimization: skip if x is too small to generate unique diffs
            # Simple loop
            valid = True
            for a in row:
                if (x - a) in used:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
            x += 1
            if x > 10000: # Safety
                return None
        rows.append(row)
    return rows

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 integers
    starting at 0, minimizing the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Initialization
    rng = random.Random(42)
    rows = _greedy_init(n_rows, k, rng)
    
    if rows is None:
        # Fallback if greedy fails (unlikely for reasonable inputs)
        # Return a very sparse valid set
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                valid = True
                for a in row:
                    if (x - a) in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                x += 1
            rows.append(row)
        
    best_rows = [r[:] for r in rows]
    best_scope = _scope_of(rows)
    
    # 2. Simulated Annealing
    # We maintain a global 'used' structure for O(1) lookups.
    # Since scope is small (< 1000 usually), we can use a set or a large bytearray.
    # Let's use a set for simplicity and safety, but optimize the diff calculation.
    
    # Pre-calculate row indices for fast access
    # rows is list of lists
    
    # Helper to get current scope
    def get_scope():
        return max(max(r) for r in rows)
    
    # Helper to compute all diffs for a row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Initialize global used set
    global_used = set()
    for row in rows:
        global_used |= get_row_diffs(row)
    
    current_scope = get_scope()
    
    # SA Parameters
    # Temperature starts high enough to allow moves, cools down
    # Since we are minimizing scope (integer), delta is integer.
    # If delta > 0 (scope increased), accept with prob exp(-delta/T).
    # If delta <= 0 (scope decreased or same), always accept (if valid).
    
    # Initial temperature
    T = max(1.0, best_scope / 10.0)
    cooling = 0.9999
    
    # Time budget for SA
    sa_end_time = start_time + 135.0 # Leave 5s for cleanup
    
    while time.time() < sa_end_time:
        # Pick a random row
        row_idx = rng.randint(0, n_rows - 1)
        row = rows[row_idx]
        
        # Pick a position to modify
        # Bias towards the last element to encourage scope reduction
        # But we must maintain increasing order.
        
        # Strategy: 
        # 1. With 50% probability, try to shrink the last element.
        # 2. With 25% probability, try to modify a random middle element.
        # 3. With 25% probability, try to expand the last element (to escape local minima).
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Shrink last element
            pos = k
            old_val = row[pos]
            min_val = row[pos-1] + 1
            if old_val <= min_val:
                continue
            # Try to reduce to a random value in [min_val, old_val - 1]
            # Or just try old_val - 1? Random jump is better for SA.
            # But large jumps might break validity. 
            # Let's try a random value in the valid range.
            # However, checking validity for a random value requires O(k) work.
            # To be efficient, we can try a few candidates.
            
            # Let's pick a candidate
            # We want to find a valid new_val < old_val.
            # If we just pick random, we might miss valid ones.
            # Better: try to move to old_val - 1, or random.
            # Let's try random in [min_val, old_val - 1].
            new_val = rng.randint(min_val, old_val - 1)
            
        elif move_type < 0.75:
            # Modify random middle element (1 to k-1)
            if k < 2:
                continue
            pos = rng.randint(1, k-1)
            old_val = row[pos]
            min_val = row[pos-1] + 1
            max_val = row[pos+1] - 1
            if min_val > max_val:
                continue
            new_val = rng.randint(min_val, max_val)
            if new_val == old_val:
                continue
                
        else:
            # Expand last element (escape)
            pos = k
            old_val = row[pos]
            # Try to increase
            max_val = old_val + 10
            if max_val < old_val + 1:
                continue
            new_val = rng.randint(old_val + 1, max_val)
            
        # Check validity of change
        # 1. Remove old diffs involving row[pos]
        # 2. Check if new diffs involving new_val are available
        # 3. If yes, commit.
        
        # Extract old diffs
        old_diffs = set()
        for i in range(k + 1):
            if i == pos:
                continue
            if i < pos:
                d = old_val - row[i]
            else:
                d = row[i] - old_val
            old_diffs.add(d)
        
        # Temporarily remove from global_used
        for d in old_diffs:
            global_used.discard(d)
            
        # Check new diffs
        new_diffs = set()
        valid = True
        for i in range(k + 1):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            if d in global_used:
                valid = False
                break
            new_diffs.add(d)
            
        if not valid:
            # Restore old diffs
            global_used.update(old_diffs)
            continue
            
        # Valid move. Calculate new scope.
        # Only the last element of this row can change the max scope if it's the max.
        # Actually, any element can be the max if we expand.
        # But for shrink, it definitely reduces or keeps same.
        
        # Compute new scope efficiently?
        # Just compute max of the modified row and global max of others.
        # Since n_rows is small, max(max(r) for r in rows) is fast.
        
        # Commit
        row[pos] = new_val
        global_used.update(new_diffs)
        
        new_scope = get_scope()
        
        # Acceptance
        delta = current_scope - new_scope
        if delta > 0:
            # Scope decreased, always accept
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
        else:
            # Scope increased (delta < 0) or same (delta == 0)
            # If delta == 0, accept to explore state space.
            if delta == 0:
                current_scope = new_scope
                # Don't update best_rows if scope is same, but we are in a new valid state.
            else:
                # delta < 0, so new_scope > current_scope
                # Accept with probability exp(delta / T)
                # Note: delta is negative here.
                if rng.random() < math.exp(delta / T):
                    current_scope = new_scope
                else:
                    # Revert
                    row[pos] = old_val
                    global_used.discard(new_diffs)
                    global_used.update(old_diffs)
        
        # Cool down
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    return best_rows

# EVOLVE-BLOCK-END
