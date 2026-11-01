import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for the 120s limit

    n_elements = k + 1
    n_diffs_per_row = k * (k + 1) // 2
    
    # Helper to check validity and get scope
    def get_scope(rows):
        if not rows:
            return float('inf')
        # Check validity implicitly by construction in moves, 
        # but for safety, we assume input rows are valid.
        # If a move creates an invalid state, we reject it or penalize it.
        return max(max(r) for r in rows)

    def is_valid_solution(rows):
        used = set()
        for r in rows:
            if len(r) != n_elements or r[0] != 0:
                return False
            # Check strictly increasing
            for i in range(1, n_elements):
                if r[i] <= r[i-1]:
                    return False
            # Check diffs
            for i in range(n_elements):
                for j in range(i+1, n_elements):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(n_elements):
                for j in range(i+1, n_elements):
                    used.add(r[j] - r[i])
        return used

    # 1. Deterministic Greedy to get initial bound
    def greedy_deterministic():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    initial_rows = greedy_deterministic()
    best_rows = initial_rows
    best_score = get_scope(best_rows)
    
    # 2. Randomized Greedy to generate diverse starts
    def greedy_randomized(window_size):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    candidates = [x]
                    # Look ahead a bit for randomization
                    for dx in range(1, window_size):
                        x_next = x + dx
                        nd_next = {x_next - a for a in row}
                        if not nd_next & used:
                            candidates.append(x_next)
                    chosen = random.choice(candidates)
                    chosen_diffs = {chosen - a for a in row}
                    row.append(chosen)
                    used |= chosen_diffs
                    x = chosen
            rows.append(row)
        return rows

    # Generate a few diverse initial solutions
    inits = [best_rows]
    for _ in range(5):
        if time.time() > deadline - 30:
            break
        r = greedy_randomized(random.randint(5, 20))
        if r and get_scope(r) < best_score:
            best_score = get_scope(r)
            best_rows = r
        inits.append(r)

    # 3. Simulated Annealing
    def simulate_annealing(initial, time_limit):
        current_rows = [r[:] for r in initial]
        current_score = get_scope(current_rows)
        best_local_rows = [r[:] for r in current_rows]
        best_local_score = current_score
        
        # Initial temperature
        # We want to accept moves that increase score by ~10% initially
        T = max(1.0, current_score * 0.1)
        T_min = 0.1
        cooling_rate = 0.999
        
        used_diffs = compute_used_diffs(current_rows)
        
        def make_move():
            """
            Returns a new state (rows, new_used_diffs) or None if no move possible.
            Moves:
            1. Local Reduction: Pick a random non-zero element, try to reduce it.
            2. Row Re-solve: Pick a random row, remove it, re-greedy it.
            """
            nonlocal used_diffs
            
            move_type = random.random()
            
            if move_type < 0.7:
                # Local Reduction / Perturbation
                # Pick a random row
                r_idx = random.randint(0, n_rows - 1)
                # Pick a random position (1 to end)
                pos = random.randint(1, n_elements - 1)
                old_val = current_rows[r_idx][pos]
                
                # Determine valid range for new value
                # Must be > current_rows[r_idx][pos-1]
                # Must be < current_rows[r_idx][pos+1] if pos < n_elements-1
                lower_bound = current_rows[r_idx][pos-1] + 1
                upper_bound = current_rows[r_idx][pos+1] - 1 if pos < n_elements - 1 else old_val + 50 # Allow expansion
                
                # We try to find a new value in [lower_bound, upper_bound]
                # To make it a stochastic move, we don't just take the smallest valid one.
                # We sample a few candidates.
                
                # First, remove diffs of old_val from used_diffs
                row = current_rows[r_idx]
                removed_diffs = set()
                for i in range(pos):
                    removed_diffs.add(old_val - row[i])
                for j in range(pos+1, n_elements):
                    removed_diffs.add(row[j] - old_val)
                
                # We need to be careful: if a diff appears in another row, we can't reuse it.
                # So the "available" set is effectively (used_diffs - removed_diffs) 
                # BUT we must check against the global used_diffs of OTHER rows.
                # It's easier to just check the new diffs against (used_diffs - removed_diffs).
                
                forbidden = used_diffs - removed_diffs
                
                # Try to find a valid new value
                # Strategy: Try a few random values in the range, or the smallest valid one.
                # For SA, exploring the neighborhood is key.
                # Let's try to pick a random valid value in a window around old_val or lower.
                
                candidate_val = None
                # Try to reduce: sample from [lower_bound, min(old_val, upper_bound)]
                # If we want to expand, sample from [old_val, upper_bound]
                
                # Let's try 10 random candidates in the valid range
                search_lo = lower_bound
                search_hi = upper_bound
                
                # Bias towards smaller values to minimize scope
                # But allow some randomness
                attempts = 0
                while attempts < 20:
                    # Sample value
                    # 80% chance to try to reduce, 20% to try nearby
                    if random.random() < 0.8:
                        target_hi = min(old_val, search_hi)
                        if target_hi >= search_lo:
                            new_val = random.randint(search_lo, target_hi)
                        else:
                            new_val = random.randint(search_lo, search_hi)
                    else:
                        new_val = random.randint(max(search_lo, old_val - 10), min(search_hi, old_val + 10))
                    
                    if new_val < search_lo or new_val > search_hi:
                        attempts += 1
                        continue
                        
                    # Check validity for this new_val
                    new_diffs = set()
                    valid = True
                    for i in range(pos):
                        d = new_val - row[i]
                        if d in forbidden:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        for j in range(pos+1, n_elements):
                            d = row[j] - new_val
                            if d in forbidden:
                                valid = False
                                break
                            new_diffs.add(d)
                            
                    if valid:
                        candidate_val = new_val
                        break
                    attempts += 1
                
                if candidate_val is None:
                    return None # No valid move found in this attempt
                
                # Apply move
                # Update used_diffs
                used_diffs -= removed_diffs
                used_diffs |= new_diffs
                
                old_row = current_rows[r_idx]
                current_rows[r_idx][pos] = candidate_val
                
                return current_rows[r_idx][pos] # Just a marker, state is modified in place

            else:
                # Row Re-solve
                r_idx = random.randint(0, n_rows - 1)
                row_to_remove = current_rows[r_idx]
                
                # Remove diffs of this row from used_diffs
                row_diffs = set()
                for i in range(n_elements):
                    for j in range(i+1, n_elements):
                        row_diffs.add(row_to_remove[j] - row_to_remove[i])
                
                used_diffs -= row_diffs
                
                # Re-greedy this row
                new_row = [0]
                x = 0
                while len(new_row) < n_elements:
                    x += 1
                    new_diffs = {x - a for a in new_row}
                    if not new_diffs & used_diffs:
                        new_row.append(x)
                        # Don't update used_diffs yet, we commit at the end
                        # Actually, we should update locally to ensure subsequent choices in this row are valid
                        # But we need to be able to revert.
                        # For simplicity, we just build the row and check if it's valid against global used_diffs
                        # Wait, the greedy step needs to know what's used.
                        # Let's simulate the greedy step with a temporary set
                        temp_used = set(used_diffs)
                        temp_row = [0]
                        tx = 0
                        success = False
                        while len(temp_row) < n_elements:
                            tx += 1
                            td = {tx - a for a in temp_row}
                            if not td & temp_used:
                                temp_row.append(tx)
                                temp_used |= td
                                success = True
                            if tx > 200: # Safety break
                                break
                        if success:
                            # Commit
                            current_rows[r_idx] = temp_row
                            used_diffs = temp_used
                            return True
                        else:
                            # Revert (do nothing, used_diffs is already reverted)
                            return None

        # Main SA Loop
        steps = 0
        while time.time() < time_limit:
            new_state = make_move()
            if new_state is not None:
                new_score = get_scope(current_rows)
                delta = new_score - current_score
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_rows = [r[:] for r in current_rows]
                else:
                    # Revert move
                    # This is tricky because make_move modifies in place.
                    # We need to undo the specific move.
                    # This implementation is flawed for easy reversion.
                    # Let's refactor make_move to return the change info or copy the state.
                    # Given time constraints, copying state is expensive but safe for small n.
                    # n_rows=7, n_elements=6. State is 42 ints. Copy is cheap.
                    pass # We need to fix the revert logic.

        # The above SA loop is problematic because of in-place modification and lack of easy revert.
        # Let's implement a cleaner version that returns new state or None.

    def simulate_annealing_v2(initial, time_limit):
        current_rows = [r[:] for r in initial]
        current_score = get_scope(current_rows)
        best_local_rows = [r[:] for r in current_rows]
        best_local_score = current_score
        
        T = max(1.0, current_score * 0.1)
        T_min = 0.01
        cooling_rate = 0.9995
        
        used_diffs = compute_used_diffs(current_rows)
        
        def try_local_move():
            """Tries to modify one element. Returns (new_rows, new_used, new_score) or None."""
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, n_elements - 1)
            old_val = current_rows[r_idx][pos]
            
            lower_bound = current_rows[r_idx][pos-1] + 1
            upper_bound = current_rows[r_idx][pos+1] - 1 if pos < n_elements - 1 else old_val + 50
            
            # Find removed diffs
            row = current_rows[r_idx]
            removed_diffs = set()
            for i in range(pos):
                removed_diffs.add(old_val - row[i])
            for j in range(pos+1, n_elements):
                removed_diffs.add(row[j] - old_val)
            
            forbidden = used_diffs - removed_diffs
            
            # Try to find a new valid value
            # Sample candidates
            for _ in range(10):
                # Bias towards reduction
                if random.random() < 0.8 and old_val > lower_bound:
                    new_val = random.randint(lower_bound, old_val)
                else:
                    new_val = random.randint(lower_bound, upper_bound)
                
                if new_val < lower_bound or new_val > upper_bound:
                    continue
                    
                new_diffs = set()
                valid = True
                for i in range(pos):
                    d = new_val - row[i]
                    if d in forbidden:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    for j in range(pos+1, n_elements):
                        d = row[j] - new_val
                        if d in forbidden:
                            valid = False
                            break
                        new_diffs.add(d)
                
                if valid:
                    # Create new state
                    new_rows = [r[:] for r in current_rows]
                    new_rows[r_idx][pos] = new_val
                    new_used = (used_diffs - removed_diffs) | new_diffs
                    new_score = get_scope(new_rows)
                    return new_rows, new_used, new_score
            return None

        def try_row_resolve():
            """Tries to re-solve a row. Returns (new_rows, new_used, new_score) or None."""
            r_idx = random.randint(0, n_rows - 1)
            row_to_remove = current_rows[r_idx]
            
            row_diffs = set()
            for i in range(n_elements):
                for j in range(i+1, n_elements):
                    row_diffs.add(row_to_remove[j] - row_to_remove[i])
            
            base_used = used_diffs - row_diffs
            
            # Greedy reconstruct
            new_row = [0]
            x = 0
            temp_used = set(base_used)
            success = False
            while len(new_row) < n_elements:
                x += 1
                nd = {x - a for a in new_row}
                if not nd & temp_used:
                    new_row.append(x)
                    temp_used |= nd
                    success = True
                if x > 200:
                    break
            
            if success:
                new_rows = [r[:] for r in current_rows]
                new_rows[r_idx] = new_row
                new_used = temp_used
                new_score = get_scope(new_rows)
                return new_rows, new_used, new_score
            return None

        while time.time() < time_limit:
            move_type = random.random()
            new_state = None
            if move_type < 0.7:
                new_state = try_local_move()
            else:
                new_state = try_row_resolve()
                
            if new_state:
                new_rows, new_used, new_score = new_state
                delta = new_score - current_score
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_rows = new_rows
                    used_diffs = new_used
                    current_score = new_score
                    
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_rows = [r[:] for r in current_rows]
                        
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_local_rows, best_local_score

    # Run SA on diverse inits
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        # Allocate remaining time proportionally or fixed chunks
        # Let's give each SA run a decent chunk of time
        # If 5 inits, ~10s each
        sa_time = min(15.0, deadline - 10 - time.time())
        if sa_time < 1.0:
            continue
            
        sa_rows, sa_score = simulate_annealing_v2(init, time.time() + sa_time)
        
        if sa_score < best_score:
            best_score = sa_score
            best_rows = sa_rows
            
    # Final Polish: Deterministic Local Search (Greedy Reduction)
    # Iterate through all elements and try to reduce them to the minimum possible valid value.
    def polish(rows):
        rows = [r[:] for r in rows]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            used_diffs = compute_used_diffs(rows)
            for r_idx in range(n_rows):
                for pos in range(n_elements - 1, 0, -1):
                    old_val = rows[r_idx][pos]
                    lower_bound = rows[r_idx][pos-1] + 1
                    if old_val == lower_bound:
                        continue
                        
                    # Remove diffs
                    removed_diffs = set()
                    for i in range(pos):
                        removed_diffs.add(old_val - rows[r_idx][i])
                    for j in range(pos+1, n_elements):
                        removed_diffs.add(rows[r_idx][j] - old_val)
                    
                    forbidden = used_diffs - removed_diffs
                    
                    # Find smallest valid new_val
                    new_val_found = None
                    for new_val in range(lower_bound, old_val):
                        new_diffs = set()
                        valid = True
                        for i in range(pos):
                            d = new_val - rows[r_idx][i]
                            if d in forbidden:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            for j in range(pos+1, n_elements):
                                d = rows[r_idx][j] - new_val
                                if d in forbidden:
                                    valid = False
                                    break
                            new_diffs.add(d)
                        
                        if valid:
                            new_val_found = new_val
                            break
                    
                    if new_val_found is not None:
                        # Apply
                        rows[r_idx][pos] = new_val_found
                        used_diffs = (used_diffs - removed_diffs) | new_diffs
                        improved = True
                        # Restart inner loop as indices/diffs changed
                        break 
                if improved:
                    break
        return rows

    final_rows = polish(best_rows)
    final_score = get_scope(final_rows)
    
    if final_score < best_score:
        best_rows = final_rows
        best_score = final_score
        
    return best_rows

if __name__ == "__main__":
    # Test with the target instance
    rows = construct(7, 5)
    print(rows)
    print("Scope:", max(max(r) for r in rows))
