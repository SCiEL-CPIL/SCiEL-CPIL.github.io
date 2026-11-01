import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = 110.0  # 10s buffer before 120s kill
    
    # 1. Greedy Initialization
    # We generate rows one by one, trying to keep them as small as possible.
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # Try to build the rest of the row
            # Heuristic: pick smallest x such that diffs are unused
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if x can be added
                # Diffs with existing elements in row
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    best_rows, best_used = greedy_init()
    best_score = max(max(r) for r in best_rows)
    
    # 2. Local Search / Simulated Annealing
    # We will try to reduce the maximum value.
    # Move: Pick a row r, pick an index i, try to move row[r][i] to a new value v.
    # To maintain validity, we need to check differences.
    
    def get_max(rows):
        return max(max(r) for r in rows)

    def is_valid_move(row, idx, new_val, used, rows):
        # new_val must not be in row
        if new_val in row:
            return False
        if new_val <= 0: # Since row starts at 0 and increases, and we keep 0 fixed? 
            # The problem says "starts at 0 and increases strictly". 
            # So row[0] must be 0. We should probably fix row[0]=0.
            # If idx == 0, return False.
            pass
        if idx == 0:
            return False
            
        # Calculate diffs to remove
        old_val = row[idx]
        diffs_to_remove = set()
        for j in range(len(row)):
            if j != idx:
                d = abs(row[idx] - row[j])
                diffs_to_remove.add(d)
        
        # Calculate diffs to add
        diffs_to_add = set()
        for j in range(len(row)):
            if j != idx:
                d = abs(new_val - row[j])
                diffs_to_add.add(d)
                
        # Check if diffs_to_add are in used (excluding diffs_to_remove)
        # Note: used contains all diffs from all rows.
        # We must ensure no diff in diffs_to_add is in used - diffs_to_remove
        # Actually, simpler: 
        # 1. Temporarily remove diffs_to_remove from used
        # 2. Check if diffs_to_add intersects used
        # 3. If yes, invalid. If no, valid.
        
        # Optimization: We can just check intersection.
        # The set `used` currently contains diffs_to_remove.
        # So we check: (diffs_to_add & used) - diffs_to_remove == empty?
        # If a diff is in both add and remove, it's fine (it's just moving within the same row? No, if it's the same diff, it means the relative distance to some other element is the same. That's allowed as long as the diff itself isn't used by *another* row).
        
        # Wait, if d is in diffs_to_add and d is in used, it might be from this row (old diff) or another row.
        # If it's from this row (old diff), it will be removed, so it's OK.
        # If it's from another row, it's NOT OK.
        
        # So: forbidden = (diffs_to_add & used) - diffs_to_remove
        if forbidden:
            return False
        return True

    def apply_move(rows, used, r, idx, new_val):
        old_val = rows[r][idx]
        # Remove old diffs
        for j in range(len(rows[r])):
            if j != idx:
                d = abs(old_val - rows[r][j])
                used.discard(d)
        
        # Update row
        rows[r][idx] = new_val
        rows[r].sort()
        
        # Add new diffs
        # Note: idx might have changed after sort. We need to find the new index of new_val?
        # Actually, we can just recompute all diffs for the row? 
        # No, we know which diffs were removed. We just add the new ones.
        # But we need to know which elements are in the row to compute new diffs.
        # The row is sorted.
        for j in range(len(rows[r])):
             # We need to skip the element that is new_val?
             # Actually, we removed diffs involving old_val.
             # Now we add diffs involving new_val.
             # We can just iterate all pairs in the row and add to used?
             # That's O(k^2). k is small (7). So O(49) per move.
             # This is safe and less error prone.
             pass
             
        # Let's just recompute all diffs for row r and add them to used.
        # But we must be careful not to double add or miss removals.
        # Better: 
        # 1. Remove all diffs of row r from used.
        # 2. Update row r.
        # 3. Add all diffs of new row r to used.
        
        # This is cleaner.
        return rows, used

    # Let's refine the move application for speed and correctness.
    def execute_move(rows, used, r, idx, new_val):
        # 1. Remove all diffs from row r
        row = rows[r]
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                used.discard(d)
        
        # 2. Update row
        row[idx] = new_val
        row.sort()
        
        # 3. Add all diffs from new row r
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                used.add(d)
                
        return rows, used

    def try_improve(rows, used, steps=10000):
        current_max = get_max(rows)
        best_max = current_max
        best_rows = [r[:] for r in rows]
        best_used = set(used)
        
        temp = 10.0
        min_temp = 0.01
        
        for _ in range(steps):
            if time.time() > deadline:
                break
                
            r = random.randint(0, n_rows - 1)
            # Don't move the first element (0)
            if len(rows[r]) < 2: continue
            idx = random.randint(1, len(rows[r]) - 1)
            
            # Try to move to a smaller value? Or random?
            # To reduce max, we should try to move large values to smaller ones.
            # Or try to move any value to a random smaller value.
            
            old_val = rows[r][idx]
            
            # Candidate new values:
            # 1. Smaller than old_val
            # 2. Random in [1, current_max]
            
            # Let's try to shrink the row.
            # Try random value in [1, old_val - 1]
            candidates = []
            if old_val > 1:
                candidates.append(random.randint(1, old_val - 1))
            # Also try swapping with another element in the same row?
            # Or moving to a random position?
            
            new_val = candidates[0] if candidates else old_val
            
            if new_val == old_val:
                continue
                
            if is_valid_move(rows[r], idx, new_val, used, rows):
                # Calculate new max
                # We don't need to fully apply to check max if we just look at the row?
                # But max is global.
                # Apply move
                rows, used = execute_move(rows, used, r, idx, new_val)
                new_max = get_max(rows)
                
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
                    best_used = set(used)
                    
                # Acceptance criterion (SA)
                # If new_max < current_max, accept.
                # If new_max > current_max, accept with prob exp(-(new_max-current_max)/temp)
                # Note: current_max is the max of the *current* state, not best.
                # We should track current state's max.
                
                # Let's simplify: Hill Climbing with restarts?
                # Or proper SA.
                
                # For now, let's just accept if it improves or equal (with prob).
                # To escape local minima, we need to accept worse moves.
                
                # Update current_max
                current_max = new_max
                temp = max(min_temp, temp * 0.999)
                
        return best_rows, best_used, best_max

    # Run multiple iterations
    iterations = 0
    while time.time() < deadline - 5.0:
        # Perturb best solution
        # Take best_rows and make a few random valid changes
        curr_rows = [r[:] for r in best_rows]
        curr_used = set(best_used)
        
        # Perturb: change 1-2 elements
        for _ in range(2):
             r = random.randint(0, n_rows - 1)
             idx = random.randint(1, len(curr_rows[r]) - 1)
             old_val = curr_rows[r][idx]
             # Try to move to random value
             for _ in range(10):
                 new_val = random.randint(1, best_score + 10)
                 if is_valid_move(curr_rows[r], idx, new_val, curr_used, curr_rows):
                     curr_rows, curr_used = execute_move(curr_rows, curr_used, r, idx, new_val)
                     break
                     
        # Local Search
        improved_rows, improved_used, improved_max = try_improve(curr_rows, curr_used, steps=2000)
        
        if improved_max < best_score:
            best_score = improved_max
            best_rows = improved_rows
            best_used = improved_used
            iterations += 1
            print(f"Improved to {best_score} at iter {iterations}")
        else:
            # If no improvement, maybe restart from greedy with different seed?
            # Or just continue.
            pass

    return best_rows

# EVOLVE-BLOCK-END
