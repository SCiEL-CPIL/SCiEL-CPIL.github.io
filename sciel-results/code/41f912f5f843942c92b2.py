import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    Uses a local search (simulated annealing style) to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def get_diffs(row):
        """Generate all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        """Check if the set of rows is valid (all diffs distinct)."""
        all_diffs = set()
        for row in rows:
            for d in get_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True

    def objective(rows):
        """Score to minimize: the largest entry."""
        max_val = 0
        for row in rows:
            if row:
                max_val = max(max_val, row[-1])
        return max_val

    # 1. Generate initial solution using greedy (fast)
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy_construct(n_rows, k)
    current_score = objective(rows)
    best_rows = [r[:] for r in rows]
    best_score = current_score
    
    # If greedy is already very good (small n_rows, k), we might still try to improve
    # But for (4,7), greedy is likely far from optimal (134 vs ~112 bound)
    
    # Local Search Strategy:
    # We try to move individual elements in a row to smaller positions.
    # Since the rows are increasing, we can only move an element $a_i$ to a smaller $v$ 
    # if $a_{i-1} < v < a_{i+1}$ (or boundaries) and the new differences don't conflict.
    
    # A more aggressive move: "Compress" a row.
    # Try to replace a row with a new valid row that has a smaller max, 
    # given the used differences of the *other* rows.
    
    def try_improve_row(row_idx, rows, used_by_others):
        """
        Attempts to find a better row for row_idx, given the differences used by others.
        Returns a new row if found, else None.
        """
        current_row = rows[row_idx]
        current_max = current_row[-1]
        
        # We want a row with max < current_max.
        # Let's search for a row with max in [current_max - 10, current_max - 1]
        # Or just try to find *any* valid row with a smaller max.
        
        # To be efficient, we can use a randomized greedy that tries to fit within a limit.
        limit = current_max - 1
        
        # Randomized search for a row that fits in [0, limit] and uses no diffs from used_by_others
        # And doesn't conflict with itself.
        
        # Since k is small (7), the number of elements is 8.
        # We can try a few random permutations or random jumps.
        
        # Simple heuristic: Try to construct a row greedily but with a cap.
        # If we hit the cap, fail.
        
        # Let's try a few random restarts of a greedy-with-cap.
        for _ in range(50): # Small number of attempts per step to keep it fast
             new_row = [0]
             x = 0
             success = True
             while len(new_row) < k + 1:
                 # Pick next x randomly between current max in new_row + 1 and limit
                 # Or just greedy with some randomness?
                 # Let's try to pick the smallest available x that doesn't conflict, 
                 # but if we get stuck, we might need backtracking. 
                 # For simplicity, let's just do a randomized greedy that picks 
                 # from a window of candidates.
                 
                 candidates = []
                 # We must be < limit
                 # We must be > last element
                 last = new_row[-1]
                 # Try a range of candidates
                 for candidate in range(last + 1, limit + 1):
                     # Check local conflicts within new_row
                     # Check global conflicts with used_by_others
                     # This is O(k^2) per candidate, which is fine for small k.
                     new_diffs = {candidate - a for a in new_row}
                     if new_diffs & used_by_others:
                         continue
                     if new_diffs & get_diffs(new_row): # Self conflict? 
                         # Wait, get_diffs(new_row) is expensive to recompute. 
                         # Let's maintain a set of current diffs for new_row.
                         pass 
                     # Actually, let's just check if candidate is valid against current new_row diffs
                     # We'll maintain a set 'cur_diffs'
                     break # This loop structure is getting messy. Let's use a helper.
             
             # Let's implement a cleaner randomized greedy for a single row with constraints.
             
        # Re-implementation of the inner loop for clarity and speed
        for _ in range(100):
            new_row = [0]
            cur_diffs = set()
            x = 0
            valid = True
            while len(new_row) < k + 1:
                # Choose next x
                # We want to stay under limit
                # Randomize the choice slightly to escape local minima
                # But we must increase strictly.
                min_next = new_row[-1] + 1
                if min_next > limit:
                    valid = False
                    break
                
                # Try a few random candidates in [min_next, limit]
                # Bias towards smaller numbers
                found = False
                # Sort candidates by some heuristic? 
                # Let's just pick a random candidate from the valid range, 
                # but prioritize smaller ones.
                # To make it "greedy-ish" but random:
                # Pick a random threshold T between min_next and limit, 
                # and try to find the smallest x in [min_next, T] that works.
                
                # For simplicity and speed, let's just try the smallest valid x first (Greedy),
                # but if we fail, we need backtracking. 
                # Since we are just trying to *find* a better row, 
                # a simple randomized greedy might not find one if it's "tight".
                
                # Alternative: Use the existing greedy but stop if we exceed limit.
                # If greedy exceeds limit, try a different randomization?
                
                # Let's stick to a simple randomized walk on the row values.
                # Actually, the "move" approach is better.
                
                # Let's abandon the "replace whole row" strategy for now as it's complex to implement efficiently 
                # without backtracking.
                
                # Instead, let's do "Single Element Shift".
                # Pick a random row, pick a random index i (not 0, not last maybe?), 
                # try to decrease a[i] by 1.
                # If valid, accept.
                
                # This is much simpler and structurally different from "construct from scratch".
                # It refines the solution.
                
                # Let's switch the strategy in the main loop to Single Element Shift.
                valid = False
                break

        return None # Placeholder, we will change the main loop logic

    # REVISED STRATEGY:
    # 1. Start with Greedy.
    # 2. Iteratively try to decrease random elements in random rows by 1.
    # 3. If the move is valid (no diff conflicts), accept it.
    # 4. If stuck, try to increase an element to allow a later decrease? (Annealing)
    #    Or just random restarts if we get stuck for too long?
    #    Given the time limit, simple hill climbing with random restarts might be enough.
    
    # Let's implement the "Decrease by 1" hill climber.
    
    def try_decrease(rows, row_idx, col_idx):
        """
        Tries to decrease rows[row_idx][col_idx] by 1.
        Returns True if successful.
        """
        if col_idx == 0:
            return False # Can't decrease 0
        
        current_val = rows[row_idx][col_idx]
        new_val = current_val - 1
        
        # Check strict increase
        if col_idx > 0 and new_val <= rows[row_idx][col_idx - 1]:
            return False
        if col_idx < len(rows[row_idx]) - 1 and new_val >= rows[row_idx][col_idx + 1]:
            return False
            
        # Calculate old diffs and new diffs
        row = rows[row_idx]
        old_diffs = set()
        new_diffs = set()
        
        # Diffs involving this element
        for i in range(len(row)):
            if i == col_idx: continue
            old_diffs.add(abs(row[col_idx] - row[i]))
            new_diffs.add(abs(new_val - row[i]))
        
        # Diffs NOT involving this element remain the same.
        # We need to check if the *new* diffs are already present in the global set.
        # And if the *old* diffs are removed, they become available.
        
        # Efficient check:
        # 1. Remove old_diffs from global_used? No, we don't have a global set passed in easily 
        #    without updating it. 
        #    Let's maintain a global set of all diffs.
        
        pass # Logic moved to main loop for state management

    # State:
    # rows: list of lists
    # global_diffs: set of all diffs currently in use
    # counts: dict of diff -> count (to handle duplicates if we were allowing them, but we aren't)
    # Actually, since all diffs must be distinct, counts are either 0 or 1.
    # So a set is sufficient.
    
    global_diffs = set()
    for r in rows:
        for d in get_diffs(r):
            global_diffs.add(d)
            
    current_score = objective(rows)
    best_score = current_score
    best_rows = [r[:] for r in rows]
    
    no_improve_count = 0
    max_no_improve = 5000 # Reset if stuck
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Random restart or perturbation
            # Let's try a random perturbation: increase a random element by 1 to "unlock"
            # or just restart the hill climbing from a slightly perturbed state.
            # For now, let's just reset the counter and try harder.
            # Or break and restart greedy?
            # Let's try a "kick": increase a random element by 1 (if valid) to change the landscape.
            kick_success = False
            for _ in range(100):
                ri = random.randint(0, n_rows - 1)
                ci = random.randint(1, len(rows[ri]) - 1) # Don't kick 0
                old_val = rows[ri][ci]
                new_val = old_val + 1
                
                # Check bounds
                if ci < len(rows[ri]) - 1 and new_val >= rows[ri][ci+1]:
                    continue
                if ci > 0 and new_val <= rows[ri][ci-1]:
                    continue
                
                # Check diffs
                row = rows[ri]
                old_diffs = {abs(old_val - row[i]) for i in range(len(row)) if i != ci}
                new_diffs = {abs(new_val - row[i]) for i in range(len(row)) if i != ci}
                
                # New diffs must not be in global_diffs
                # Old diffs will be removed, so they are "free"
                # But wait, if a new_diff is in old_diffs, it's fine?
                # No, new_diffs are the *new* set of diffs for this row.
                # The diffs *not* involving ci are unchanged.
                # The diffs involving ci are replaced.
                
                # So:
                # 1. Remove old_diffs from global_diffs
                # 2. Check if any new_diff is in global_diffs (after removal)
                # 3. If yes, revert.
                # 4. If no, add new_diffs to global_diffs and accept.
                
                # Optimization: 
                # If new_diffs & (global_diffs - old_diffs) is not empty, invalid.
                
                conflict = new_diffs & (global_diffs - old_diffs)
                if not conflict:
                    # Accept kick
                    global_diffs -= old_diffs
                    global_diffs |= new_diffs
                    rows[ri][ci] = new_val
                    kick_success = True
                    no_improve_count = 0 # Reset counter on successful move
                    break
            
            if not kick_success:
                # If we can't even kick, we might be in a dead end.
                # Restart from greedy?
                # Or just break and return best?
                # Let's try to restart from a new greedy solution to escape local minima.
                # But greedy is deterministic. 
                # Let's use randomized greedy for restart.
                
                def randomized_greedy(n_rows, k, seed=None):
                    if seed is not None: random.seed(seed)
                    used = set()
                    res_rows = []
                    for _ in range(n_rows):
                        row = [0]
                        x = 0
                        while len(row) < k + 1:
                            x += 1
                            new = {x - a for a in row}
                            if not new & used:
                                row.append(x)
                                used |= new
                        res_rows.append(row)
                    return res_rows
                
                # Perturb the greedy by adding random noise to the selection?
                # Hard to do with the simple greedy.
                # Let's just restart the hill climbing from the current best?
                # No, that's where we came from.
                
                # Let's try a different strategy: 
                # If stuck, try to "swap" two elements in a row?
                # Or just accept that we might be stuck and return best.
                
                # To avoid infinite loop, let's break if we've been stuck for a long time.
                if no_improve_count > 20000:
                    break
                no_improve_count += 1
                continue

        # Normal Hill Climbing Step: Try to decrease
        ri = random.randint(0, n_rows - 1)
        ci = random.randint(1, len(rows[ri]) - 1) # Can't decrease index 0
        
        old_val = rows[ri][ci]
        new_val = old_val - 1
        
        # Check strict increase
        if ci > 0 and new_val <= rows[ri][ci - 1]:
            no_improve_count += 1
            continue
        if ci < len(rows[ri]) - 1 and new_val >= rows[ri][ci + 1]:
            no_improve_count += 1
            continue
            
        row = rows[ri]
        old_diffs = {abs(old_val - row[i]) for i in range(len(row)) if i != ci}
        new_diffs = {abs(new_val - row[i]) for i in range(len(row)) if i != ci}
        
        # Check conflicts
        # The new diffs must not be in the global set, EXCEPT for the ones we are removing (old_diffs).
        # Actually, if a new_diff is in old_diffs, it means we are re-using a diff that was just freed up?
        # No, old_diffs are removed from global. new_diffs are added.
        # If new_diff == old_diff, it's fine.
        # If new_diff is in global_diffs AND not in old_diffs, it's a conflict.
        
        conflict = new_diffs & (global_diffs - old_diffs)
        
        if not conflict:
            # Accept
            global_diffs -= old_diffs
            global_diffs |= new_diffs
            rows[ri][ci] = new_val
            
            # Update score
            new_score = objective(rows)
            if new_score < best_score:
                best_score = new_score
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
    # Return the best valid solution found
    # Ensure best_rows is valid (it should be, as we only made valid moves)
    # But just in case, verify.
    if is_valid(best_rows):
        return best_rows
    else:
        # Fallback to greedy if something went wrong
        return greedy_construct(n_rows, k)
