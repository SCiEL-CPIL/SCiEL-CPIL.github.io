"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Helper to compute max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Helper to check validity and return set of differences
    def get_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    diffs.add(d)
        return diffs

    def is_valid(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    # 1. Initial Valid Solution via Greedy
    def pure_greedy():
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

    best_rows = pure_greedy()
    best_max = get_max_entry(best_rows)
    best_diffs = get_diffs(best_rows)
    
    # 2. Local Search: Conflict-Driven Variable Selection (CDVS)
    # Strategy: 
    # We want to reduce max_entry. 
    # We look for a row and an entry that is "large" (close to max).
    # We try to move it to a smaller value.
    # The new value must not create any new conflicts.
    # If it does, we identify the conflicting difference and try to "fix" it 
    # by moving the *other* row that uses that difference.
    
    # We maintain a set of "forbidden" differences for the current candidate move.
    
    # To speed up, we precompute for each difference d, which rows contain it.
    # But rows change. So we recompute on the fly or maintain incrementally.
    # Given small n_rows (4) and k (7), recomputation is cheap.
    
    # Let's try a simpler but effective heuristic:
    # "Compress the largest entry".
    # Identify the row with the largest last element.
    # Try to reduce the last element of that row.
    # If blocked, try to reduce the second last, etc.
    # If a move is blocked by a conflict with difference d in row R2, 
    # try to move an element in R2 to free up d.
    
    # This is a recursive local search.
    
    def try_move(rows, r_idx, c_idx, new_val):
        """
        Attempt to set rows[r_idx][c_idx] to new_val.
        Returns (success, new_rows, new_diffs, conflict_info)
        conflict_info: if failed, returns (conflicting_diff, conflicting_row_idx)
        """
        # Check bounds
        if c_idx > 0 and new_val <= rows[r_idx][c_idx - 1]:
            return False, None, None, None
        if c_idx < k and new_val >= rows[r_idx][c_idx + 1]:
            return False, None, None, None
        
        # Calculate new differences for this row
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        # Simulate new row
        new_row = old_row[:]
        new_row[c_idx] = new_val
        
        new_diffs = set()
        for i in range(len(new_row)):
            for j in range(i+1, len(new_row)):
                new_diffs.add(new_row[j] - new_row[i])
        
        # Differences that are removed
        removed_diffs = old_diffs - new_diffs
        # Differences that are added
        added_diffs = new_diffs - old_diffs
        
        # Check if any added_diff is already in global_diffs (excluding those in removed_diffs? No, global_diffs is the set of ALL diffs in ALL rows)
        # Wait, global_diffs contains diffs from ALL rows.
        # If we change row r, we remove old_diffs from the "pool" of used diffs for that row, but they might still be used by other rows.
        # Actually, the constraint is that ALL diffs across ALL rows must be distinct.
        # So if an added_diff d is present in ANY other row, it's a conflict.
        # If d was in old_diffs, it was already "used" by this row. But since all diffs must be distinct, d was NOT used by any other row.
        # So if d is in added_diffs, we need to check if d is in the set of diffs of OTHER rows.
        
        # Let's maintain a global counter or set.
        # For simplicity in this helper, we pass the current global set of diffs.
        # But we need to know which diffs are "safe" to use.
        # A diff is safe if it is not in the set of diffs of any other row.
        
        # To do this efficiently, we need to know for each diff d, how many rows contain it.
        # Since valid solution has count 1 for all d, if we move a diff out of row R1, it becomes "free" (count 0) unless another row has it (which is impossible in valid state).
        # So, if we move a diff d from row R1 to row R2, we must ensure d is not in R3 (for R3 != R1, R2).
        
        # Let's implement a helper to check conflicts against other rows.
        
        for d in added_diffs:
            # Check if d is in any other row
            for r_other in range(n_rows):
                if r_other == r_idx:
                    continue
                # Check if d is in rows[r_other]
                row_other = rows[r_other]
                found = False
                for i in range(len(row_other)):
                    for j in range(i+1, len(row_other)):
                        if row_other[j] - row_other[i] == d:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return False, None, None, (d, r_other)
        
        # If no conflicts, apply move
        new_rows = [row[:] for row in rows]
        new_rows[r_idx] = new_row
        
        # Update global diffs (conceptually, we return the new rows, caller updates global state)
        return True, new_rows, None, None

    # Iterative Local Search Loop
    # We will try to reduce the maximum entry.
    # We identify the "bottleneck" entry: the one that is currently the max.
    # We try to reduce it.
    
    # To make it more robust, we can randomize which "large" entry we try to reduce.
    
    for _ in range(100000): # Max iterations
        if time.time() > deadline:
            break
        
        # Find the row and index of the current maximum
        max_val = -1
        max_r = -1
        max_c = -1
        for r in range(n_rows):
            for c in range(k+1):
                if rows_val := best_rows[r][c]:
                    if rows_val > max_val:
                        max_val = rows_val
                        max_r = r
                        max_c = c
                        # Note: 0 is at c=0, so we start c from 1 usually, but max is always > 0
        
        if max_r == -1:
            break # All zeros? Impossible for k>0
        
        # Try to reduce best_rows[max_r][max_c]
        # We try values from (best_rows[max_r][max_c] - 1) down to (best_rows[max_r][max_c-1] + 1)
        # But checking every value is slow. 
        # Instead, let's look at the conflicts that PREVENT a reduction.
        
        # Actually, a simpler heuristic: 
        # Pick the largest entry. Try to move it to a random smaller valid slot.
        # If that fails, try to "make space" by moving other entries.
        
        # Let's try a "Swap" strategy.
        # Find two entries in different rows that can swap values to reduce the max.
        # Or, find an entry in the max row that can be moved to a smaller value.
        
        # Let's stick to: Try to reduce the specific max entry.
        current_val = best_rows[max_r][max_c]
        min_allowed = best_rows[max_r][max_c - 1] + 1 if max_c > 0 else 1
        max_allowed = best_rows[max_r][max_c + 1] - 1 if max_c < k else current_val
        
        # We want to move to a value < current_val.
        # Let's pick a random target in [min_allowed, current_val - 1]
        if min_allowed >= current_val:
            # Cannot reduce this entry further without violating local ordering
            # Try another entry? Or break?
            # Let's try to reduce the second largest, etc.
            # For simplicity, if we can't reduce the max, we try to reduce the entry BEFORE the max?
            # No, let's just pick a random row and a random index (not 0) and try to reduce it.
            
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)
            current_val = best_rows[r_idx][c_idx]
            min_allowed = best_rows[r_idx][c_idx - 1] + 1
            max_allowed = best_rows[r_idx][c_idx + 1] - 1 if c_idx < k else current_val
            
            if min_allowed >= current_val:
                continue
                
            # Try to move to a smaller value
            # We try a few random candidates
            found_move = False
            for _ in range(20):
                target_val = random.randint(min_allowed, current_val - 1)
                success, new_rows, _, conflict_info = try_move(best_rows, r_idx, c_idx, target_val)
                if success:
                    best_rows = new_rows
                    found_move = True
                    # Update best_max
                    new_max = get_max_entry(best_rows)
                    if new_max < best_max:
                        best_max = new_max
                    break
                else:
                    # Conflict detected. 
                    # We could use conflict_info to guide a move in the other row.
                    # For now, we just try another random target.
                    pass
            
            if not found_move:
                # Try a "kick" move: move an entry in the conflicting row to a random valid spot
                # This allows us to escape local minima.
                # Identify a random row and entry and try to move it to a random valid spot (not necessarily smaller).
                # This is a random walk to escape.
                r_kick = random.randint(0, n_rows - 1)
                c_kick = random.randint(1, k)
                current_val_kick = best_rows[r_kick][c_kick]
                min_kick = best_rows[r_kick][c_kick - 1] + 1
                max_kick = best_rows[r_kick][c_kick + 1] - 1 if c_kick < k else best_max
                
                # Pick a random valid value
                candidates = []
                for v in range(min_kick, max_kick + 1):
                    if v != current_val_kick:
                        # Quick check if v is valid locally
                        candidates.append(v)
                
                if candidates:
                    target_val_kick = random.choice(candidates)
                    success, new_rows, _, _ = try_move(best_rows, r_kick, c_kick, target_val_kick)
                    if success:
                        best_rows = new_rows
                        # Don't update best_max here as it might increase, but it allows future reductions
                        # However, we must ensure we don't permanently increase the scope.
                        # Let's only accept kick moves if they don't increase the max.
                        new_max_kick = get_max_entry(new_rows)
                        if new_max_kick <= best_max:
                            best_rows = new_rows
                        else:
                            # Reject kick if it increases max
                            pass
        else:
            # We can reduce the max entry
            found_move = False
            for _ in range(50):
                # Try to move to a smaller value
                # Bias towards smaller values?
                target_val = random.randint(min_allowed, current_val - 1)
                success, new_rows, _, conflict_info = try_move(best_rows, max_r, max_c, target_val)
                if success:
                    best_rows = new_rows
                    new_max = get_max_entry(best_rows)
                    if new_max < best_max:
                        best_max = new_max
                    found_move = True
                    break
                else:
                    # Conflict. 
                    # Try to resolve conflict by moving the other row?
                    # This is complex. Let's just try another target.
                    pass
            
            if not found_move:
                # Stuck on reducing the max. 
                # Perform a random kick elsewhere to change the landscape.
                r_kick = random.randint(0, n_rows - 1)
                c_kick = random.randint(1, k)
                current_val_kick = best_rows[r_kick][c_kick]
                min_kick = best_rows[r_kick][c_kick - 1] + 1
                max_kick = best_rows[r_kick][c_kick + 1] - 1 if c_kick < k else best_max
                
                candidates = [v for v in range(min_kick, max_kick + 1) if v != current_val_kick]
                if candidates:
                    target_val_kick = random.choice(candidates)
                    success, new_rows, _, _ = try_move(best_rows, r_kick, c_kick, target_val_kick)
                    if success:
                        new_max_kick = get_max_entry(new_rows)
                        if new_max_kick <= best_max:
                            best_rows = new_rows

    return best_rows
# EVOLVE-BLOCK-END
