import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 140  # Safety margin for 150s limit
    
    # Helper to calculate differences of a row
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to check if a set of rows is valid (all diffs unique across all rows)
    # Returns (is_valid, all_diffs)
    def check_validity(rows):
        all_diffs = set()
        for row in rows:
            row_diffs = get_diffs(row)
            # Check internal validity (should always be true for strictly increasing ints? 
            # No, e.g., 0,1,3 diffs: 1,3,2. 0,2,4 diffs: 2,4,2 -> invalid internally)
            # But our construction ensures internal validity.
            if len(row_diffs) != len(row) * (len(row) - 1) // 2:
                return False, set()
            if all_diffs & row_diffs:
                return False, set()
            all_diffs |= row_diffs
        return True, all_diffs

    # Helper to get scope (max entry)
    def get_scope(rows):
        return max(max(row) for row in rows)

    # 1. Randomized Greedy Construction
    def randomized_greedy():
        for _ in range(50):
            rows = []
            used_diffs = set()
            success = True
            for _ in range(n_rows):
                row = [0]
                # We want to keep the row compact, so we try small increments
                # But we must avoid conflicts
                while len(row) < k + 1:
                    existing_diffs = get_diffs(row)
                    found = False
                    # Search for next element x > row[-1]
                    # To encourage compactness, we check x from row[-1]+1 upwards
                    # But randomizing the start or step can help escape local traps
                    # Let's try a window of candidates
                    candidates = []
                    # Heuristic: The next element shouldn't be too far if we want low scope
                    # But we don't know the target scope yet.
                    # Let's just scan upwards.
                    for x in range(row[-1] + 1, row[-1] + 50):
                        new_diffs = {x - a for a in row}
                        # Check internal conflicts
                        if new_diffs & existing_diffs:
                            continue
                        # Check global conflicts
                        if new_diffs & used_diffs:
                            continue
                        candidates.append(x)
                        if len(candidates) >= 5:
                            break
                    
                    if not candidates:
                        # If stuck, we might need to restart this row or whole process
                        # For now, break and restart whole greedy
                        success = False
                        break
                    
                    # Pick randomly from candidates to introduce diversity
                    # Bias towards smaller x to keep scope low?
                    # random.choice(candidates) is uniform.
                    # Let's weight towards smaller x.
                    weights = [1.0 / (x - row[-1]) for x in candidates]
                    x = random.choices(candidates, weights=weights, k=1)[0]
                    
                    row.append(x)
                    used_diffs |= {x - a for a in row[:-1]}
                    # Note: existing_diffs is updated implicitly in next loop iteration via get_diffs(row)
                
                if not success:
                    break
                rows.append(row)
            
            if success:
                # Verify validity
                is_valid, _ = check_validity(rows)
                if is_valid:
                    return rows
        return None

    # 2. Local Search
    def local_search(rows, max_time=5.0):
        start_ls = time.time()
        rows = [r[:] for r in rows]
        scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = scope
        
        # Precompute diffs for current state
        row_diffs = [get_diffs(row) for row in rows]
        all_diffs = set()
        for d in row_diffs:
            all_diffs |= d

        while time.time() - start_ls < max_time:
            # Pick a random row
            i = random.randint(0, n_rows - 1)
            # Pick a random position to modify (1 to k)
            # We prioritize modifying the last element (largest) or elements near the max
            # to reduce scope.
            # Let's pick position j uniformly first.
            j = random.randint(1, k)
            
            old_val = rows[i][j]
            old_row = rows[i][:]
            old_row_diffs = row_diffs[i]
            
            # Determine search range for new_val
            # Must be > rows[i][j-1] and < rows[i][j+1] (if j < k)
            low = rows[i][j-1] + 1
            high = rows[i][j+1] - 1 if j < k else scope + 20 # Allow some growth
            
            # If we are trying to reduce scope, and j is the last element, 
            # we should try values lower than old_val.
            # If j < k, we are constrained by neighbors.
            
            found_improvement = False
            # Try a few random candidates in the valid range
            for _ in range(20):
                # Bias towards reducing value if possible
                if j == k and old_val == scope:
                    # Try to reduce
                    if low < old_val:
                        new_val = random.randint(low, old_val - 1)
                    else:
                        continue
                else:
                    # Random in range
                    if low > high:
                        continue
                    new_val = random.randint(low, high)
                
                if new_val == old_val:
                    continue
                
                # Check bounds
                if new_val <= rows[i][j-1]:
                    continue
                if j < k and new_val >= rows[i][j+1]:
                    continue
                
                # Construct new row
                new_row = old_row[:]
                new_row[j] = new_val
                
                # Check internal validity of new_row
                # Since we only changed one element, we can check diffs involving j
                # But it's safer to just compute all diffs for this small row
                new_row_diffs = get_diffs(new_row)
                if len(new_row_diffs) != len(new_row) * (len(new_row) - 1) // 2:
                    continue
                
                # Check conflicts with other rows
                # Other diffs = all_diffs - old_row_diffs
                other_diffs = all_diffs - old_row_diffs
                if new_row_diffs & other_diffs:
                    continue
                
                # Valid move
                # Calculate new scope
                # Only row i changed
                new_max_row_i = max(new_row)
                # Global max is max(max of other rows, new_max_row_i)
                # We can approximate by just checking if this reduces the global max
                # Or just compute get_scope for the new state
                # To be fast, let's just compute max of all rows
                # But we can optimize: if new_max_row_i < old_val and old_val was the unique max, scope decreases.
                # For simplicity and correctness, compute scope.
                temp_rows = rows[:]
                temp_rows[i] = new_row
                new_scope = get_scope(temp_rows)
                
                if new_scope < scope:
                    # Accept improvement
                    rows = temp_rows
                    row_diffs[i] = new_row_diffs
                    all_diffs = other_diffs | new_row_diffs
                    scope = new_scope
                    found_improvement = True
                    
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [r[:] for r in rows]
                    break # Restart local search from this new state? Or continue?
                    # Usually in ILS, we accept and continue. 
                    # But if we break, we restart the while loop with new state.
                
                # If no improvement, we might accept a worse move with some probability (Simulated Annealing style)
                # But for now, let's stick to Hill Climbing for simplicity and speed, 
                # relying on ILS restarts for exploration.
                # If we don't break, we try next candidate.
            
            if not found_improvement:
                # No improvement found in this iteration
                # We could do a perturbation here to escape local minima
                # But the outer ILS loop handles restarts.
                # To make local search more effective, we might want to accept neutral moves?
                # Let's just continue.
                pass

        return best_rows

    # 3. Main ILS Loop
    best_rows = randomized_greedy()
    if best_rows is None:
        # Fallback: Deterministic greedy
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs) and len(new_diffs) == len(row):
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
    
    best_scope = get_scope(best_rows)
    
    # ILS Loop
    while time.time() - start_time < deadline:
        # Perturb best_rows
        perturbed_rows = [r[:] for r in best_rows]
        # Randomly move a few elements
        for _ in range(3):
            i = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            old_val = perturbed_rows[i][j]
            # Try to move to a random valid position
            low = perturbed_rows[i][j-1] + 1
            high = perturbed_rows[i][j+1] - 1 if j < k else old_val + 10
            if low <= high:
                new_val = random.randint(low, high)
                # Check validity quickly
                new_row = perturbed_rows[i][:]
                new_row[j] = new_val
                new_diffs = get_diffs(new_row)
                if len(new_diffs) == len(new_row) * (len(new_row) - 1) // 2:
                    # Check global conflict? 
                    # For perturbation, we might allow invalid states? 
                    # No, local search expects valid states.
                    # So we must ensure the perturbation results in a valid state.
                    # This is hard. 
                    # Alternative: Perturb by swapping or small shifts that are likely valid?
                    # Or just restart from scratch if perturbation fails?
                    # Let's just restart from scratch if we get stuck.
                    pass
        
        # Actually, a simpler ILS strategy:
        # 1. Start from a random valid solution (or best)
        # 2. Run local search
        # 3. If stuck, restart from a new random valid solution or a perturbed version of the best.
        
        # Let's do this:
        # 1. Get a random valid solution (fast)
        # 2. Run local search on it
        # 3. Keep the best found
        
        # To avoid repeating the same random solutions, we can use a seed or just rely on randomness.
        
        # Optimization: If we have time, run multiple local searches from different starts.
        
        current_rows = randomized_greedy()
        if current_rows is None:
            continue
            
        # Run local search on current_rows
        improved_rows = local_search(current_rows, max_time=min(5.0, deadline - (time.time() - start_time)))
        
        current_scope = get_scope(improved_rows)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in improved_rows]
            
            # If we hit the theoretical lower bound or close, we can stop early?
            # Lower bound for k=5, n=7 is 105.
            # If best_scope == 105, we are optimal.
            if best_scope == 105:
                break

    return best_rows
