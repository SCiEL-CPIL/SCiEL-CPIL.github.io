import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _reconstruct_row(other_used, k, current_row, time_limit=0.5):
    """
    Reconstruct a single row to minimize its max entry, given the set of differences
    already used by other rows.
    Uses Simulated Annealing for the single row.
    """
    n = k + 1
    # Start with the current row or a greedy init
    # Greedy init is often better for reconstruction as it starts compact
    row = [0]
    x = 0
    used_local = set()
    # Check against other_used
    while len(row) < n:
        x += 1
        new_diffs = {x - a for a in row}
        # Must not collide with local diffs or other_used
        if not (new_diffs & used_local) and not (new_diffs & other_used):
            row.append(x)
            used_local |= new_diffs
        else:
            # If greedy gets stuck or is inefficient, we might need to backtrack,
            # but for reconstruction, a simple greedy with small steps is usually okay.
            # However, greedy might get stuck in a local optimum of high max.
            # We will refine with SA.
            pass
    
    # If greedy didn't fill the row (unlikely if other_used is sparse enough for a valid solution to exist),
    # we might have a problem. But assuming the global state is valid, a solution for this row exists.
    # Actually, the above greedy loop above is flawed if it doesn't fill the row.
    # Let's just use the current_row as the starting point for SA, as it is known valid.
    row = current_row[:]
    
    start_time = time.time()
    T = 50.0
    T_min = 1.0
    cool = 0.99
    best_row = row[:]
    best_max = max(row)
    
    # Precompute diffs for current row
    curr_diffs = _row_diffs_set(row)
    
    while time.time() - start_time < time_limit:
        # Pick a position to modify
        pi = random.randrange(1, n)
        old_val = row[pi]
        
        # Generate candidates
        min_allowed = row[pi - 1] + 1
        max_allowed = (row[pi + 1] - 1) if pi < n - 1 else best_max + 10
        
        # Bias towards smaller values
        candidates = []
        # Decreasing
        for d in range(1, min(20, old_val - min_allowed + 1)):
            candidates.append(old_val - d)
        # Increasing
        for d in range(1, min(10, max_allowed - old_val + 1)):
            candidates.append(old_val + d)
        
        if not candidates:
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= row[pi - 1] or (pi < n - 1 and new_val >= row[pi + 1]):
                continue
                
            # Test validity
            # Update row temporarily
            row[pi] = new_val
            
            # Compute new diffs
            new_diffs = set()
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                row[pi] = old_val
                continue
                
            # Check against other_used
            conflict = False
            for d in new_diffs:
                if d in other_used:
                    conflict = True
                    break
            
            if conflict:
                row[pi] = old_val
                continue
                
            # Valid move
            new_max = max(row)
            delta = new_max - best_max # Note: we compare against best_max of this row, not global
            # Actually, we want to minimize this row's max.
            
            accept = False
            if new_max < best_max:
                accept = True
            else:
                if T > T_min:
                    prob = math.exp(-max(0, new_max - best_max) / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update best
                if new_max < best_max:
                    best_max = new_max
                    best_row = row[:]
                # Keep row as is (accepted)
                # No need to update curr_diffs explicitly for next step unless we use it for delta calc, 
                # but we recalc diffs every time.
                moved = True
                break
            else:
                row[pi] = old_val
        
        if not moved:
            row[pi] = old_val # Ensure restore if loop finished without move
            
        T = max(T_min, T * cool)
        
    return best_row

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # 2. Iterative Row Reconstruction (Lloyd's-like)
    # We iterate through rows, removing their diffs, reconstructing them to minimize max,
    # and adding them back.
    
    # To make it more effective, we can do multiple passes.
    # Also, we can introduce randomness in the order of rows.
    
    row_order = list(range(n_rows))
    
    # Time budget for each row reconstruction
    # Total time ~ 115s.
    # If we do 10 passes, 10 * n_rows reconstructions.
    # n_rows is 7. 70 reconstructions.
    # 115 / 70 ~ 1.6s per reconstruction.
    # Let's aim for ~1.0s per reconstruction to be safe, allowing for overhead.
    
    recon_time = 0.8
    
    pass_count = 0
    max_passes = 50 # Cap passes, time limit will stop us
    
    while time.time() < deadline and pass_count < max_passes:
        random.shuffle(row_order)
        pass_improved = False
        
        for ri in row_order:
            if time.time() > deadline - 5.0: # Stop if close to deadline
                break
                
            # Remove current row's diffs from global set
            curr_diffs = _row_diffs_set(rows[ri])
            other_used = all_used - curr_diffs
            
            # Reconstruct row
            new_row = _reconstruct_row(other_used, k, rows[ri], recon_time)
            
            # Check if new row is valid (it should be by construction, but just in case)
            new_diffs = _row_diffs_set(new_row)
            if new_diffs & other_used:
                # Fallback to old row if reconstruction failed validity (shouldn't happen)
                new_row = rows[ri][:]
                new_diffs = curr_diffs
            
            # Calculate new global max
            # We need to update the global max.
            # old_max_row = max(rows[ri])
            # new_max_row = max(new_row)
            # The global max is max(max(rows[j]) for j != ri, new_max_row)
            
            # To be efficient, we can track row_maxes
            # But for simplicity and correctness, let's just compute if it looks promising
            # Or maintain a list of row maxes.
            
            # Let's maintain row_maxes for speed
            # We don't have row_maxes in scope, let's just compute global max if new_row is smaller
            # Actually, if new_row max is smaller than old_row max, global max might decrease.
            # If new_row max is larger, global max might increase.
            
            # Let's just try it and check.
            old_row = rows[ri]
            new_row_max = max(new_row)
            old_row_max = max(old_row)
            
            # Potential improvement if new_row_max < old_row_max
            # Or if it allows other rows to shrink later.
            
            # Accept the new row if it doesn't increase the global max too much?
            # In Lloyd's, we accept the assignment.
            # Here, we should accept if it improves the objective (global max) or is neutral.
            # If it increases global max, we might still accept it if it helps future rows?
            # Given the time limit, let's be conservative: only accept if global max doesn't increase.
            
            # To check global max without full recompute:
            # If new_row_max <= old_row_max:
            #   Global max can only stay same or decrease.
            #   We should accept.
            # If new_row_max > old_row_max:
            #   Global max might increase.
            #   We should probably reject, unless we are in an "exploration" phase.
            # Let's stick to "accept if global max does not increase".
            
            # Efficient check:
            # If new_row_max < old_row_max:
            #   It's an improvement for this row.
            #   Global max = max(max(rows[j] for j!=ri), new_row_max)
            #   Old Global Max = max(max(rows[j] for j!=ri), old_row_max)
            #   Since new_row_max < old_row_max, New Global Max <= Old Global Max.
            #   So we accept.
            
            # If new_row_max == old_row_max:
            #   Global max stays same. Accept to shuffle.
            
            # If new_row_max > old_row_max:
            #   Global max might increase. Reject.
            
            if new_row_max <= old_row_max:
                # Update state
                rows[ri] = new_row
                all_used -= curr_diffs
                all_used |= new_diffs
                
                # Update best if improved
                # We need to know the new global max.
                # We can compute it.
                new_global = _global_max(rows)
                if new_global < best_val:
                    best_val = new_global
                    best_rows = [r[:] for r in rows]
                    pass_improved = True
                elif new_global == best_val:
                    # Neutral move, keep it to allow future improvements
                    pass
            else:
                # Reject, keep old row
                pass
        
        pass_count += 1
        
        # If no improvement in a pass, we might want to restart or change strategy
        # But with row reconstruction, neutral moves can help.
        # If stuck for many passes, we can try a full SA restart.
        # For now, just continue.
        
        # Safety: if we are very close to deadline, break
        if time.time() > deadline - 10.0:
             break

    return best_rows

# EVOLVE-BLOCK-END
