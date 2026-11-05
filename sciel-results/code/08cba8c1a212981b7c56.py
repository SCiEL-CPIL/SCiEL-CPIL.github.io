import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # If k is small, we can be very thorough.
    # If k is large, we need efficient heuristics.
    
    # 1. Generate an initial valid solution using greedy method
    # This ensures we always have a valid output.
    best_rows = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_rows)
    
    # If k is very small, we might even be able to find the optimal quickly.
    # For (4,7), greedy might give ~150-200, we want 134.
    
    # 2. Local Search to minimize scope
    # We will run multiple restarts of local search if time permits.
    
    rng = random.Random(42)
    
    # We keep the best solution found so far
    global_best_rows = best_rows
    global_best_scope = best_scope
    
    while time.time() < deadline:
        # Initialize current state with a copy of best found so far
        # Or restart from greedy? Restarting from greedy might be better to explore different basins.
        # Let's alternate: sometimes start from best, sometimes from fresh greedy.
        
        if rng.random() < 0.5:
            current_rows = [row[:] for row in global_best_rows]
        else:
            current_rows = greedy_construct(n_rows, k)
            
        current_scope = max(max(row) for row in current_rows)
        
        # Prepare used differences set
        used_diffs = set()
        for row in current_rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        # Local search loop
        no_improve_count = 0
        max_no_improve = 500 # Adjust based on size
        
        while time.time() < deadline:
            # Pick a random move
            # We want to reduce values.
            # Pick a random row and a random index j > 0
            r = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[r][j]
            
            # Determine valid range for new value
            # Must be > current_rows[r][j-1]
            # Must be < current_rows[r][j+1] if j < k, else no upper bound (but we want to reduce, so < old_val)
            lower_bound = current_rows[r][j-1] + 1
            upper_bound = old_val - 1 # We only try to reduce
            
            if lower_bound > upper_bound:
                # Can't reduce this element directly.
                # Maybe we can increase it to allow others to decrease? 
                # For now, stick to reduction moves to minimize scope.
                # If we can't reduce any, we are stuck.
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                continue
            
            # Try to find a valid new value in [lower_bound, upper_bound]
            # To be efficient, we can sample a few values or check from lower_bound upwards.
            # Checking from lower_bound upwards is better for minimization.
            
            # Optimization: If the range is large, sampling might be better.
            # But for "compression", we want the smallest possible.
            # Let's try a few candidates: the smallest possible, and some random ones.
            
            candidates = []
            # Always try the smallest possible
            candidates.append(lower_bound)
            # Add some random candidates in the range
            for _ in range(5):
                if lower_bound < upper_bound:
                    candidates.append(rng.randint(lower_bound, upper_bound))
            
            found_valid = False
            for new_val in candidates:
                if new_val >= old_val:
                    continue
                
                # Check validity
                # 1. Remove old diffs involving current_rows[r][j]
                old_diffs_to_remove = set()
                for i in range(k + 1):
                    if i == j: continue
                    d = abs(current_rows[r][j] - current_rows[r][i])
                    old_diffs_to_remove.add(d)
                
                # Check if new diffs conflict with existing used_diffs (excluding old_diffs_to_remove)
                # And check if new diffs conflict with each other (within the row, excluding old ones)
                
                # Actually, it's easier to:
                # 1. Calculate new diffs for the row with new_val
                # 2. Check if any new diff is in (used_diffs - old_diffs_to_remove)
                # 3. Check if any new diff is in (new_diffs - {new_val - current_rows[r][i] for i != j})? No, just check against global set.
                
                new_diffs = set()
                valid = True
                for i in range(k + 1):
                    if i == j: continue
                    d = abs(new_val - current_rows[r][i])
                    new_diffs.add(d)
                    
                    # Check against other diffs in the same row (that don't involve j)
                    # This is implicitly handled if we check against the global set 
                    # AFTER removing old diffs and BEFORE adding new diffs.
                    # But we must ensure new_diffs are distinct among themselves?
                    # Yes, because they are differences from a single element to others in a strictly increasing sequence, 
                    # they are naturally distinct if the sequence is strictly increasing.
                    # Wait, if a_i < a_j < a_k, then a_j-a_i != a_k-a_j unless a_i+a_k=2a_j.
                    # But a_j-a_i and a_k-a_j are both in new_diffs.
                    # Are they distinct? 
                    # Example: 0, 2, 4. Diffs: 2, 2, 4. Conflict!
                    # So we MUST check if new_diffs has duplicates.
                    if len(new_diffs) < (k): # Should be k distinct values if no internal conflict
                        # Check for internal duplicates in new_diffs
                        # Actually, just checking len(new_diffs) == k is enough?
                        # The number of other elements is k. So there are k differences.
                        # If any are equal, the set size will be < k.
                        pass 
                
                if len(new_diffs) != k:
                    valid = False
                else:
                    # Check against global used diffs
                    # We need to check if any d in new_diffs is in (used_diffs - old_diffs_to_remove)
                    remaining_used = used_diffs - old_diffs_to_remove
                    if new_diffs & remaining_used:
                        valid = False
                
                if valid:
                    # Apply move
                    # Update used_diffs
                    used_diffs -= old_diffs_to_remove
                    used_diffs |= new_diffs
                    
                    current_rows[r][j] = new_val
                    
                    # Update scope
                    new_scope = max(max(row) for row in current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        no_improve_count = 0
                    else:
                        # Scope didn't change, but we made progress in compression?
                        # We accept it to potentially allow further reductions.
                        no_improve_count = 0 
                    
                    found_valid = True
                    break
            
            if not found_valid:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        # End of local search for this instance
        if current_scope < global_best_scope:
            global_best_scope = current_scope
            global_best_rows = [row[:] for row in current_rows]
            
        # If we found the theoretical lower bound, we can stop?
        # Lower bound is n * k(k+1)/2
        # But this is a weak bound.
        # For (4,7), LB = 4 * 28 = 112. Best known 134.
        
        # Check if we should restart or break
        # If time is running out, break.
        
    return global_best_rows

def greedy_construct(n_rows, k):
    """
    Constructs a valid DTS using a greedy approach.
    Fills rows one by one, left to right, picking the smallest valid number.
    """
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last = row[-1]
            x = last + 1
            while True:
                # Check if x creates any conflicts
                conflict = False
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used_diffs.add(x - a)
                    break
                x += 1
        rows.append(row)
        
    return rows
