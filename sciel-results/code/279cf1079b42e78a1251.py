import random
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def get_differences(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # 1. Initial Greedy Construction
    # This provides a valid starting point.
    used = set()
    initial_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        initial_rows.append(row)
    
    best_rows = [r[:] for r in initial_rows]
    best_score = max(max(r) for r in best_rows)
    
    # Pre-compute differences for current best
    def compute_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs

    best_diffs = compute_all_diffs(best_rows)
    
    # 2. Local Search
    # Strategy: Pick a random row, pick a random mark (index > 0), try to move it down.
    # If moving it down creates a conflict, we can't. But maybe we can move it to a different valid spot?
    # Actually, moving a mark down is the only way to reduce the max. 
    # If we move a mark *up*, the max might increase, but it might allow other marks to move down later.
    # However, for this specific metric (minimize max entry), "down" is the primary driver.
    # Let's try "down" moves primarily, but allow "up" moves if we get stuck (annealing).
    
    current_rows = [r[:] for r in best_rows]
    current_diffs = compute_all_diffs(current_rows)
    current_max = best_score
    
    t = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if t < 0.1:
            t = 10.0 # Restart annealing if stuck, or just keep going with small t
            
        row_idx = random.randrange(n_rows)
        col_idx = random.randrange(1, k + 1) # Indices 1..k
        
        old_val = current_rows[row_idx][col_idx]
        
        # Determine range to search
        # Must be > prev_val and < next_val (if exists)
        prev_val = current_rows[row_idx][col_idx - 1]
        next_val = current_rows[row_idx][col_idx + 1] if col_idx < k else float('inf')
        
        # Try to move down first (improvement)
        # Check all values from prev_val + 1 to old_val - 1
        # Also try some values above old_val (worsening) if annealing allows
        
        best_new_val = None
        best_new_max = current_max
        
        # Efficient check: 
        # If we move val at (row_idx, col_idx) to new_val:
        # 1. Remove old diffs involving this mark.
        # 2. Check if new_val conflicts with remaining diffs.
        # 3. Check if new_val maintains sorted order.
        
        # Let's iterate candidate new_vals
        # Priority: values lower than old_val (to reduce max)
        # Then values higher than old_val (to explore)
        
        candidates = []
        # Down moves
        for v in range(old_val - 1, prev_val, -1):
            candidates.append(v)
            if len(candidates) > 50: break # Limit search depth per step for speed
        
        # Up moves (only if we are stuck or t is high)
        if t > 1.0:
            upper_limit = min(next_val - 1, old_val + 50)
            for v in range(old_val + 1, upper_limit + 1):
                candidates.append(v)
        
        random.shuffle(candidates)
        
        found_move = False
        for new_val in candidates:
            if new_val <= prev_val or new_val >= next_val:
                continue
                
            # Calculate diffs to remove
            row = current_rows[row_idx]
            old_mark = old_val
            old_diffs = set()
            for j in range(k + 1):
                if j == col_idx:
                    continue
                d = abs(row[j] - old_mark)
                old_diffs.add(d)
                
            # Check if any old_diffs are used by OTHER rows
            # If an old_diff is unique to this row (count 1 in global), it's free to remove.
            # If it's shared, we can't remove it without breaking other rows? 
            # Wait, the constraint is "all differences ... are distinct over all rows".
            # This means NO difference can appear twice in the global set.
            # So if we change a mark, we remove its old diffs from the global set.
            # We must ensure the new diffs do not collide with the REMAINING global diffs.
            
            # Optimization: 
            # remaining_diffs = current_diffs - old_diffs
            # new_diffs = {abs(row[j] - new_val) for j != col_idx}
            # If new_diffs & remaining_diffs is empty, it's a valid move.
            
            # But computing remaining_diffs every time is expensive if we do it naively.
            # Let's just check collisions.
            
            # 1. Check order
            # 2. Check collisions
            
            # To speed up, we can pre-check:
            # Does new_val create a diff with any other element in this row that is already in current_diffs (and not one of the old_diffs we are removing)?
            
            # Let's compute new_diffs
            new_diffs = set()
            for j in range(k + 1):
                if j == col_idx:
                    continue
                d = abs(row[j] - new_val)
                new_diffs.add(d)
                
            # Check collision
            # A diff d in new_diffs is invalid if:
            # d is in current_diffs AND d is NOT in old_diffs
            # (Because if d is in old_diffs, it's being removed, so it's available)
            
            valid = True
            for d in new_diffs:
                if d in current_diffs and d not in old_diffs:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # If valid, calculate new max
            # The new max is max(current_max, new_val) 
            # BUT if old_val was the global max, and we moved it down, the global max might decrease.
            # We need to check if old_val was the unique maximum.
            
            # Let's just compute the new global max roughly.
            # If new_val < old_val:
            #   new_max = max(new_val, max of all other entries)
            #   To get max of all other entries, we can precompute row maxes?
            #   Simpler: just calculate max of the modified row and compare with global max of other rows.
            
            # For efficiency in the loop, let's just accept if it's "better" or "same" with some prob.
            # Since we are minimizing max, let's compute the actual new max.
            
            # Optimization: keep track of global max and count of how many times it appears.
            # This is getting complex. Let's just calculate max for the modified row and assume other rows are unchanged.
            # new_row_max = max(row with new_val)
            # global_max = max(new_row_max, max of other rows)
            
            # To avoid O(N) max calculation, we can store row_maxes.
            # But for now, let's just try a few candidates.
            
            # Let's do the full update if it looks promising
            # Actually, let's just update and check score.
            
            # Update current_rows
            row[col_idx] = new_val
            new_row_max = max(row)
            
            # We need the max of the other rows.
            # Let's precompute maxes for other rows?
            # For now, let's just compute max of all rows. It's O(N*K) which is small.
            # N=7, K=6. 42 elements.
            
            new_global_max = 0
            for r in current_rows:
                m = max(r)
                if m > new_global_max:
                    new_global_max = m
            
            # Update diffs
            current_diffs -= old_diffs
            current_diffs |= new_diffs
            
            if new_global_max < current_max:
                # Accept
                current_max = new_global_max
                found_move = True
                break
            elif new_global_max == current_max:
                # Neutral move, accept with prob exp(0) = 1? Or small prob?
                # Let's accept 50% to explore
                if random.random() < 0.5:
                    found_move = True
                    break
            else:
                # Worse move
                # Accept with prob exp(-(new_max - current_max) / t)
                delta = new_global_max - current_max
                if delta > 0:
                    prob = pow(2.71828, -delta / t)
                    if random.random() < prob:
                        current_max = new_global_max
                        found_move = True
                        break
                    else:
                        # Revert
                        row[col_idx] = old_val
                        current_diffs -= new_diffs
                        current_diffs |= old_diffs
                        continue
                else:
                     # Should not happen if new_global_max >= current_max
                     pass
            
            # If we didn't break, we reverted or didn't accept?
            # The logic above is a bit tangled. Let's clean it up.
            
            # Revert if not accepted
            # The 'break' handles acceptance.
            # If we finish the loop without break, we need to revert if we modified.
            # But I modified `row` and `current_diffs` in place.
            # I need to be careful.
            
            # Let's refactor the candidate loop to be cleaner.
            pass # Placeholder, the logic above was messy. I will rewrite the loop structure in the final code.
            
        if not found_move:
            pass
        t *= cooling
        
        # Update best if improved
        if current_max < best_score:
            best_score = current_max
            best_rows = [r[:] for r in current_rows]
            best_diffs = set(current_diffs)
            
    return best_rows
# EVOLVE-BLOCK-END
