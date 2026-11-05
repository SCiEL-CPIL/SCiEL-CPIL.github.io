import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    best_rows = None
    best_scope = float('inf')
    
    def get_row_diffs(row):
        """Compute set of all pairwise differences for a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_row(row, used_diffs_excluding_self):
        """Check if a row is internally valid and doesn't conflict with other rows."""
        if not row[0] == 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        diffs = get_row_diffs(row)
        if len(diffs) != len(row) * (len(row) - 1) // 2:
            return False
        if diffs & used_diffs_excluding_self:
            return False
        return True

    def compute_used_diffs(rows, exclude_idx=-1):
        """Compute the set of all differences used by all rows except the one at exclude_idx."""
        used = set()
        for i, r in enumerate(rows):
            if i == exclude_idx:
                continue
            used.update(get_row_diffs(r))
        return used

    def greedy_construct(max_scope):
        """
        Construct rows greedily with a scope limit.
        Returns list of rows or None if failed.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            valid = True
            for pos in range(1, k + 1):
                last = row[-1]
                # Find the smallest x > last such that diffs are valid
                # x must be <= max_scope
                found = False
                for x in range(last + 1, max_scope + 1):
                    # New diffs introduced by x
                    new_diffs = {x - a for a in row}
                    # Check internal uniqueness
                    if len(new_diffs) != len(row):
                        continue
                    # Check global uniqueness
                    if new_diffs & used:
                        continue
                    # Valid
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    valid = False
                    break
            if valid:
                rows.append(row)
            else:
                return None
        return rows

    def local_search(rows):
        """
        Improve rows by moving elements to reduce the maximum scope.
        Returns improved rows and the new max scope.
        """
        current_rows = [list(r) for r in rows]
        current_max = max(max(r) for r in current_rows)
        
        # Precompute used diffs for each row index to speed up checks?
        # Or just recompute the "others" set when a row is touched.
        # Since n_rows is small (7), recomputing others is O(6 * 15) = 90 ops, which is fast.
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Shuffle row order to break symmetry
            row_order = list(range(n_rows))
            random.shuffle(row_order)
            
            for i in row_order:
                if time.time() - start_time > time_limit:
                    break
                row = current_rows[i]
                # Get diffs of all other rows
                used_others = compute_used_diffs(current_rows, exclude_idx=i)
                
                # Try to improve this row
                # Iterate positions from last to first (largest values first)
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    min_val = row[pos-1] + 1
                    # We want to try smaller values first to reduce scope
                    # But we might need to try larger values to escape local minima?
                    # For now, focus on reduction. If stuck, we might need a more complex search.
                    
                    # Try values from min_val to current_val - 1
                    # Optimization: only try values that might reduce the global max
                    # If current_val is not the global max, reducing it might not help the objective function (min max).
                    # But it might free up differences for other rows to be reduced later.
                    
                    # Let's try a random subset of valid smaller values to avoid O(N^2) checks
                    # Actually, for k=5, the range is small.
                    
                    candidates = list(range(min_val, current_val))
                    random.shuffle(candidates)
                    
                    for v in candidates:
                        # Check bounds
                        if v > current_val: # We are iterating smaller, so this is false
                            continue
                        # Check upper bound if pos < len(row)-1
                        if pos < len(row) - 1 and v >= row[pos+1]:
                            continue
                        
                        new_row = list(row)
                        new_row[pos] = v
                        
                        # Check validity
                        if is_valid_row(new_row, used_others):
                            # Accept
                            current_rows[i] = new_row
                            new_max = max(max(r) for r in current_rows)
                            if new_max < current_max:
                                current_max = new_max
                            improved = True
                            # Update used_others for subsequent checks in this row?
                            # No, we break out of the position loop after a successful move?
                            # Or continue to see if we can improve further in this row?
                            # Let's break to restart the row order shuffle logic or just continue to next pos?
                            # Restarting is safer to avoid stale used_others.
                            break
                    
                    if improved:
                        # We modified the row, so used_others for this row is stale for the next pos?
                        # Actually, if we move pos, the diffs change.
                        # It's safer to break out of the position loop and let the outer loop handle it.
                        break
            
        return current_rows, current_max

    def iterated_local_search(rows):
        """
        Run local search, then perturb, then run local search again.
        """
        current_rows = [list(r) for r in rows]
        current_max = max(max(r) for r in current_rows)
        
        while time.time() - start_time < time_limit:
            # Local search
            improved_rows, improved_max = local_search(current_rows)
            if improved_max < current_max:
                current_rows = improved_rows
                current_max = improved_max
            
            # Perturbation (Kick)
            # Randomly change a few elements to escape local minima
            # Choose a random row and a random position
            # Change the value to a random valid value (might increase scope temporarily)
            if time.time() - start_time > time_limit - 5:
                break
                
            for _ in range(3): # 3 random moves
                i = random.randint(0, n_rows - 1)
                pos = random.randint(1, len(current_rows[i]) - 1)
                row = current_rows[i]
                used_others = compute_used_diffs(current_rows, exclude_idx=i)
                
                # Find a random valid value
                min_v = row[pos-1] + 1
                max_v = row[pos+1] - 1 if pos < len(row) - 1 else current_max + 10
                # Limit search space for speed
                max_v = min(max_v, current_max + 20)
                
                valids = []
                # Sample some values
                for v in range(min_v, max_v + 1):
                    new_row = list(row)
                    new_row[pos] = v
                    if is_valid_row(new_row, used_others):
                        valids.append(v)
                
                if valids:
                    new_val = random.choice(valids)
                    current_rows[i][pos] = new_val
                    # Don't check global max validity here, just accept the state
                    # The next local search will fix it or reject it if it's worse?
                    # Actually, local search only accepts if it reduces max.
                    # If the perturbation increases max, local search won't accept it unless it leads to a better max later.
                    # This is the "iterated" part.
            
            # Check if we improved
            # local_search already returned the best found in that pass.
            # We need to re-evaluate if the perturbed state leads to a better result.
            # The local_search call above operated on the *pre-perturbation* state?
            # No, the loop is:
            # 1. Local search on current
            # 2. Perturb current
            # 3. Loop back to 1.
            # So the next iteration of the while loop will local search the perturbed state.
            
    # Main loop
    # 1. Try to find an initial solution with a scope slightly above the theoretical minimum (105)
    #    Best known is 112. Let's start with 112.
    
    initial_scope = 112
    # Try greedy first
    init_rows = greedy_construct(initial_scope)
    
    if init_rows is None:
        # Try a slightly larger scope
        init_rows = greedy_construct(120)
        
    if init_rows is None:
        # Fallback to a very loose greedy
        init_rows = greedy_construct(200)
        
    if init_rows is not None:
        best_rows = init_rows
        best_scope = max(max(r) for r in best_rows)
        
        # Apply ILS
        improved_rows, improved_scope = iterated_local_search(best_rows)
        if improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
            
    # If we still have time, try multiple restarts with different random seeds/strategies
    while time.time() - start_time < time_limit - 10:
        # Randomized Greedy with scope cap
        scope_cap = best_scope if best_scope < float('inf') else 120
        # Try to construct with a slightly tighter cap to force better packing
        target_scope = max(105, scope_cap - 5)
        
        rows = greedy_construct(target_scope)
        if rows is None:
            rows = greedy_construct(scope_cap)
            
        if rows is not None:
            scope = max(max(r) for r in rows)
            # Local search
            improved_rows, improved_scope = local_search(rows)
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = improved_rows
                
            # If we hit the theoretical lower bound, we can stop
            if best_scope <= 105:
                break

    if best_rows is None:
        # Absolute fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
