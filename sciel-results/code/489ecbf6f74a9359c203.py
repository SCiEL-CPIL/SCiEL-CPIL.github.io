import random
import time
import math

def construct(n_rows, k):
    row_len = k + 1
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Bitmask for used differences. 
    # Max difference is likely < 200. 
    # Python ints can handle arbitrary size, so 1 << 200 is fine.
    
    def get_used_mask(rows):
        mask = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    mask |= (1 << (row[j] - row[i]))
        return mask

    def is_valid_diff(diff, mask):
        if diff <= 0:
            return False
        return (mask & (1 << diff)) == 0

    def greedy_init(randomize=False):
        rows = [[0] for _ in range(n_rows)]
        mask = 0
        # We need to add k points to each row
        # Total additions: n_rows * k
        for step in range(n_rows * k):
            # Select row
            if randomize:
                # Pick a random row that is not full
                candidates = [i for i in range(n_rows) if len(rows[i]) < row_len]
                i = random.choice(candidates)
            else:
                # Pick row with smallest length, tie-break by smallest last val
                i = min(range(n_rows), key=lambda idx: (len(rows[idx]), rows[idx][-1]))
            
            if len(rows[i]) >= row_len:
                continue
                
            current_row = rows[i]
            last_val = current_row[-1]
            
            # Find valid x
            # Heuristic: Try small x first, but maybe randomize slightly
            candidates = []
            x = last_val + 1
            # Limit search depth to avoid infinite loop if stuck (shouldn't happen with greedy)
            limit = last_val + 200 
            while x <= limit:
                # Check conflicts
                valid = True
                for v in current_row:
                    d = x - v
                    if not is_valid_diff(d, mask):
                        valid = False
                        break
                if valid:
                    candidates.append(x)
                    if not randomize or len(candidates) >= 5:
                        break
                x += 1
            
            if not candidates:
                # Fallback: just keep incrementing until found
                while True:
                    valid = True
                    for v in current_row:
                        d = x - v
                        if not is_valid_diff(d, mask):
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                        break
                    x += 1
            
            new_val = random.choice(candidates) if randomize else candidates[0]
            current_row.append(new_val)
            for v in current_row[:-1]:
                mask |= (1 << (new_val - v))
                
        return rows, mask, max(max(r) for r in rows)

    start_time = time.time()
    timeout = 110
    
    # 1. Get a baseline
    baseline_rows, baseline_mask, baseline_cost = greedy_init(randomize=False)
    best_rows = [r[:] for r in baseline_rows]
    best_mask = baseline_mask
    best_cost = baseline_cost
    
    # 2. Try a few randomized greedy inits
    for _ in range(10):
        if time.time() - start_time > timeout - 20: break
        r, m, c = greedy_init(randomize=True)
        if c < best_cost:
            best_rows = [r[:] for r in r]
            best_mask = m
            best_cost = c
            
    # 3. Local Search to improve best_cost
    # Strategy: Try to reduce the max element by shifting points down.
    # We iterate through all rows and all positions (except 0).
    # We try to decrease the value.
    # If valid, we accept.
    # We repeat this until no improvement or time limit.
    
    current_rows = [r[:] for r in best_rows]
    current_mask = best_mask
    current_cost = best_cost
    
    improved = True
    while improved and (time.time() - start_time < timeout - 5):
        improved = False
        # Randomize the order of operations to escape local minima?
        # Or just do a full pass.
        for i in range(n_rows):
            if time.time() - start_time > timeout - 5: break
            row = current_rows[i]
            # Process from right to left? Or left to right?
            # Reducing a value affects differences with all subsequent values.
            # It's safer to process from right to left? No, reducing a[i] affects a[j]-a[i] for j>i.
            # If we reduce a[i], we must ensure new diffs are valid.
            # Let's try all positions.
            for j in range(len(row) - 1, 0, -1):
                old_val = row[j]
                # Try to reduce old_val
                # New val must be > row[j-1]
                # And < old_val
                new_val = old_val - 1
                while new_val > row[j-1]:
                    # Check if new_val creates conflicts
                    # Diffs with previous elements: new_val - row[i] for i < j
                    # Diffs with next elements: row[k] - new_val for k > j
                    # Note: Diffs among other elements are unchanged.
                    
                    valid = True
                    # Check diffs with i < j
                    for p in range(j):
                        d = new_val - row[p]
                        if not is_valid_diff(d, current_mask):
                            valid = False
                            break
                    if not valid:
                        new_val -= 1
                        continue
                        
                    # Check diffs with k > j
                    # The old diffs were row[k] - old_val.
                    # The new diffs are row[k] - new_val.
                    # We need to remove old diffs from mask, check new, then re-add?
                    # Or just check if new diffs are in mask (excluding the old ones we are about to replace).
                    
                    # To be safe and simple, let's construct the new mask for this row's changes?
                    # That's expensive.
                    
                    # Alternative: 
                    # 1. Remove all diffs involving old_val from mask.
                    # 2. Check if new diffs are in mask.
                    # 3. If yes, add new diffs to mask and accept.
                    # 4. If no, restore old diffs and try next new_val.
                    
                    # Step 1: Remove old diffs
                    old_diffs = []
                    for p in range(j):
                        d = old_val - row[p]
                        old_diffs.append(d)
                        current_mask &= ~(1 << d)
                    for q in range(j + 1, len(row)):
                        d = row[q] - old_val
                        old_diffs.append(d)
                        current_mask &= ~(1 << d)
                        
                    # Step 2: Check new diffs
                    valid = True
                    new_diffs = []
                    for p in range(j):
                        d = new_val - row[p]
                        if not is_valid_diff(d, current_mask):
                            valid = False
                            break
                        new_diffs.append(d)
                    if not valid:
                        # Restore
                        for d in old_diffs:
                            current_mask |= (1 << d)
                        new_val -= 1
                        continue
                        
                    for q in range(j + 1, len(row)):
                        d = row[q] - new_val
                        if not is_valid_diff(d, current_mask):
                            valid = False
                            break
                        new_diffs.append(d)
                        
                    if valid:
                        # Accept move
                        row[j] = new_val
                        for d in new_diffs:
                            current_mask |= (1 << d)
                        # Update cost
                        new_cost = max(max(r) for r in current_rows)
                        if new_cost < current_cost:
                            current_cost = new_cost
                            improved = True
                            # Don't break, keep trying to reduce further?
                            # Yes, loop continues to next j or i
                        # If new_cost == current_cost, we might still want to move to enable future moves?
                        # For now, only accept if it reduces cost or is neutral?
                        # Let's only accept if it reduces cost to avoid cycles.
                        # Actually, neutral moves can help. But let's keep it simple.
                    else:
                        # Restore
                        for d in old_diffs:
                            current_mask |= (1 << d)
                        new_val -= 1
                        continue
                        
                    # If we accepted, we should probably restart the inner loop or just continue?
                    # If we changed row[j], the subsequent j's in this row are affected?
                    # No, we are iterating j downwards. 
                    # If we change row[j], it affects diffs with row[k] for k>j.
                    # But we already processed k>j? No, we are at j.
                    # The loop goes from len-1 down to 1.
                    # So when we are at j, we have already "optimized" j+1, j+2, etc.
                    # Changing row[j] affects diffs with j+1...
                    # So we might invalidate the "optimality" of j+1?
                    # But we only accept if valid.
                    # We should probably break and restart the row optimization if we make a move?
                    # Or just continue.
                    break # Break inner while loop, move to next j? Or next i?
                    
        # End while loop
    
    # Final check
    if current_cost < best_cost:
        best_rows = [r[:] for r in current_rows]
        best_cost = current_cost
        
    return best_rows
