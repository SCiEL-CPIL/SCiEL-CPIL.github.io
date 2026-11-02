"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # --- Helper Functions ---
    
    def compute_diffs(rows):
        """Compute the set of all pairwise differences."""
        diffs = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    diffs.add(row[j] - row[i])
        return diffs

    def get_scope(rows):
        """Get the maximum entry across all rows."""
        return max(max(row) for row in rows)

    def greedy_init(n_rows, k):
        """Greedy initialization: each row starts at 0 and takes the smallest next mark 
        whose differences to the earlier marks of the row are all still unused by any row."""
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
        return rows

    def try_move(rows, used_diffs, r, j, new_val):
        """
        Attempt to move row[r][j] to new_val.
        Returns True if valid and applied, False otherwise.
        Modifies rows and used_diffs in place.
        """
        old_val = rows[r][j]
        if old_val == new_val:
            return True
        
        # Check bounds and ordering
        if new_val < 0:
            return False
        if j > 0 and new_val <= rows[r][j - 1]:
            return False
        if j < len(rows[r]) - 1 and new_val >= rows[r][j + 1]:
            return False
        
        # Remove old differences involving index j
        # Note: Differences are positive: row[b] - row[a] where b > a.
        # If j is the larger index (b), diff = row[j] - row[i] for i < j
        # If j is the smaller index (a), diff = row[i] - row[j] for i > j
        old_diffs_to_remove = set()
        for i in range(len(rows[r])):
            if i == j:
                continue
            if i < j:
                old_diffs_to_remove.add(rows[r][j] - rows[r][i])
            else:
                old_diffs_to_remove.add(rows[r][i] - rows[r][j])
        
        # Check if new differences conflict with existing used_diffs
        # We must exclude the diffs we are about to remove from the conflict check
        # effectively: (new_diffs - old_diffs) & (used_diffs - old_diffs) == empty
        # Simpler: Remove old from used, check new against used, if ok add new, else revert.
        
        # Optimization: Check conflicts first without modifying state if possible
        # But we need to know which diffs are "free".
        # Let's do the remove-check-add pattern.
        
        # 1. Remove old diffs from global set
        for d in old_diffs_to_remove:
            if d in used_diffs:
                used_diffs.remove(d)
        
        # 2. Generate new diffs
        new_diffs_to_add = set()
        for i in range(len(rows[r])):
            if i == j:
                continue
            if i < j:
                new_diffs_to_add.add(new_val - rows[r][i])
            else:
                new_diffs_to_add.add(rows[r][i] - new_val)
        
        # 3. Check for conflicts with remaining used_diffs
        if new_diffs_to_add & used_diffs:
            # Conflict found, revert
            for d in old_diffs_to_remove:
                used_diffs.add(d)
            return False
        
        # 4. Apply change
        rows[r][j] = new_val
        for d in new_diffs_to_add:
            used_diffs.add(d)
            
        return True

    # --- Main Logic ---

    # 1. Greedy Initialization
    rows = greedy_init(n_rows, k)
    used_diffs = compute_diffs(rows)
    best_rows = [row[:] for row in rows]
    best_scope = get_scope(best_rows)
    
    # 2. Local Search with Iterated Perturbation
    # We try to minimize the scope.
    # Strategy: 
    #   - Identify the current scope S.
    #   - Try to move any mark equal to S to a smaller value.
    #   - If successful, S might decrease.
    #   - If stuck, try random moves to escape local minima.
    
    current_rows = [row[:] for row in rows]
    current_used = set(used_diffs)
    current_scope = best_scope
    
    max_iterations = 10000
    iter_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        # --- Local Search Phase ---
        improved = False
        local_iter = 0
        max_local_iter = 500 # Limit local search steps per restart
        
        while local_iter < max_local_iter and time.time() < deadline:
            local_iter += 1
            
            # Find all marks that are equal to current_scope
            # Moving these is the only way to reduce scope immediately
            # However, we also need to move other marks to create space.
            
            # Strategy: 
            # 1. Pick a random row.
            # 2. Pick a random index j (preferably high value or critical).
            # 3. Try to move it to a smaller value if possible.
            
            r = random.randint(0, n_rows - 1)
            row = current_rows[r]
            
            # Bias towards indices with larger values
            # We want to reduce the max.
            # Let's pick an index that is part of the current maximums or just random.
            # To reduce scope, we MUST move a mark that is currently equal to the max scope.
            # If no mark is at max scope (impossible by definition of scope), we are in a local min wrt scope?
            # Actually, scope is max of all entries. If we move a non-max entry, scope stays same.
            # We only improve if we move a max entry to a smaller value.
            
            max_val = current_scope
            candidates = [i for i in range(1, len(row)) if row[i] == max_val]
            
            if candidates:
                # We must move one of these to reduce scope
                j = random.choice(candidates)
                # Try to move to a smaller value
                # The new value must be > row[j-1] and < row[j+1] (if exists)
                # And must not create duplicate diffs.
                
                # Try values in descending order from row[j]-1 down to row[j-1]+1
                # Or random sampling?
                # Let's try a few random smaller values first, then systematic.
                
                low_bound = row[j-1] + 1
                high_bound = row[j] - 1
                
                if low_bound <= high_bound:
                    # Try to find a valid smaller value
                    # Randomize the order of attempts to avoid bias
                    possible_vals = list(range(low_bound, high_bound + 1))
                    random.shuffle(possible_vals)
                    
                    found = False
                    for new_val in possible_vals:
                        if try_move(current_rows, current_used, r, j, new_val):
                            # Check if scope decreased
                            new_scope = get_scope(current_rows)
                            if new_scope < current_scope:
                                current_scope = new_scope
                                improved = True
                                if current_scope < best_scope:
                                    best_scope = current_scope
                                    best_rows = [row[:] for row in current_rows]
                            break
                    # If no valid smaller value found, the move failed (reverted by try_move)
                    # We might try to move it to a larger value? No, that increases scope.
                    # We might try to move OTHER marks to free up space for this one?
                    # That's complex. For now, if we can't move the max down, we are stuck locally.
            
            else:
                # No mark at max scope? This shouldn't happen if current_scope is correctly tracked.
                # But maybe we moved a max mark down, and now scope is lower, but we didn't update current_scope?
                # get_scope is O(N). Let's update it if we made a move that didn't change scope?
                # Actually, if we moved a non-max mark, scope is unchanged.
                # If we moved a max mark down but it's still the max (e.g., 100 -> 99, but another is 100), scope is 100.
                # So candidates list might be empty if we moved the *only* max mark down?
                # No, if we moved the only max mark down, scope decreases, so we update current_scope.
                # So if candidates is empty, it means current_scope is not present in any row?
                # That implies current_scope is stale.
                pass
            
            # If we couldn't improve scope, we might be stuck.
            # We can try to move random marks to perturb the state.
            # But if we don't reduce scope, we don't improve best.
            
            # To escape local minima where we can't move the max down:
            # We need to "shuffle" other marks to free up differences.
            # Let's do a random perturbation if we haven't improved in a while.
            # For simplicity in this loop, if we didn't improve in this local step, 
            # and we are in a "stuck" state (no candidates or no valid moves for candidates), 
            # we break out of local search to restart.
            
            # Heuristic: If we tried to move a candidate and failed, or no candidates, 
            # we might be stuck. 
            # Let's just break if we didn't improve in this specific iteration 
            # and we've done some local steps.
            if not improved and local_iter > 20:
                break

        # --- Perturbation Phase ---
        # If we broke out of local search without reducing scope further,
        # perturb the current solution to escape the local minimum.
        
        # Perturb 2-5 random marks
        num_perturb = random.randint(2, 5)
        for _ in range(num_perturb):
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, len(current_rows[r]) - 1)
            # Shift value by small amount
            old_val = current_rows[r][j]
            # Try a random shift
            for _ in range(10):
                delta = random.randint(-5, 5)
                if delta == 0: continue
                new_val = old_val + delta
                if try_move(current_rows, current_used, r, j, new_val):
                    # Update scope if necessary
                    # Note: try_move doesn't update scope.
                    # We recalc scope after perturbation?
                    # It's expensive. 
                    # But perturbation usually increases or keeps scope.
                    # We just need to ensure validity.
                    break
        
        # Recalculate scope for current state
        current_scope = get_scope(current_rows)
        
        # If perturbation made it worse, that's fine, we explore.
        # But we must ensure we don't run out of time.
        
        # Optional: If current scope is much worse than best, we might want to revert to best?
        # Iterated Local Search usually keeps the current state as the start for next local search.
        # If we want to be safe, we can restart from best if we've wandered too far.
        # Let's stick to standard ILS: keep current state.
        
        # Check time
        if time.time() > deadline - 5:
            break

    return best_rows
# EVOLVE-BLOCK-END
