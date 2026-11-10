import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Parameters for the specific task: n_rows=4, k=7
    # Each row has k+1 = 8 elements
    # Number of differences per row = 8*7/2 = 28
    # Total differences = 4 * 28 = 112
    
    # The theoretical lower bound for the scope (max element) is 112.
    # The best known is 134.
    
    def get_row_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        """
        Returns (is_valid, scope, conflicts)
        conflicts is the number of duplicate differences found.
        """
        used_diffs = set()
        valid = True
        scope = 0
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        valid = False
                        conflicts += 1
                    else:
                        used_diffs.add(d)
            if max(row) > scope:
                scope = max(row)
        return valid, scope, conflicts

    def greedy_construct():
        """
        Generates a valid configuration using a greedy approach.
        This is used as the initial state for Simulated Annealing.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(new_diffs)
            rows.append(row)
        return rows

    # Initialize with greedy solution
    current_rows = greedy_construct()
    valid, current_scope, current_conflicts = is_valid(current_rows)
    
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    best_conflicts = current_conflicts
    best_valid = valid

    # Cost function:
    # Primary: minimize conflicts (must be 0 for valid)
    # Secondary: minimize scope
    # We want to find a valid solution with the smallest scope.
    # A good cost function for SA:
    # If conflicts > 0: cost = conflicts * 1000 + scope
    # If conflicts == 0: cost = scope
    # This ensures we prioritize validity over scope initially, but once valid, we minimize scope.
    # However, we might get stuck in a valid state with high scope and never escape to a lower scope valid state
    # if the path requires going through invalid states.
    # To handle this, we can use a softer penalty or allow moving to invalid states if it reduces scope significantly?
    # No, the goal is "minimize largest entry" subject to validity.
    # So we MUST be valid.
    # Let's define cost = conflicts * 10000 + scope.
    # This heavily penalizes invalid states.
    
    def get_cost(rows):
        v, s, c = is_valid(rows)
        return c * 10000 + s

    current_cost = get_cost(current_rows)
    
    # Simulated Annealing Parameters
    T = 50.0
    alpha = 0.995
    min_T = 0.1
    steps = 0
    
    while time.time() < deadline - 1:
        # Generate Move
        # Pick random row and index (excluding index 0 which is always 0)
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, k)
        
        # Generate new value
        # We want to perturb the value.
        # The value must be strictly greater than the previous element and strictly less than the next.
        old_val = current_rows[r_idx][i_idx]
        
        # Determine bounds for the new value
        lower_bound = current_rows[r_idx][i_idx - 1] + 1
        if i_idx < k:
            upper_bound = current_rows[r_idx][i_idx + 1] - 1
        else:
            # For the last element, we can allow it to grow, but maybe limit the growth
            upper_bound = old_val + 20
            
        if lower_bound > upper_bound:
            # This can happen if the row is very tight.
            # In this case, we might need to move other elements too, but for now, skip.
            # Or, we can try to swap with neighbors?
            continue
        
        # Randomly pick a new value in the range
        # Bias towards small changes
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        if new_val < lower_bound:
            new_val = lower_bound
        elif new_val > upper_bound:
            new_val = upper_bound
            
        if new_val == old_val:
            continue

        # Apply move
        current_rows[r_idx][i_idx] = new_val
        
        # Calculate new cost
        new_valid, new_scope, new_conflicts = is_valid(current_rows)
        new_cost = get_cost(current_rows)
        
        delta_cost = new_cost - current_cost
        
        # Accept/Reject
        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            current_scope = new_scope
            current_conflicts = new_conflicts
            current_valid = new_valid
            
            # Update best
            # We only update best if it is valid.
            if current_valid and current_scope < best_scope:
                best_scope = current_scope
                best_conflicts = 0
                best_valid = True
                best_rows = [r[:] for r in current_rows]
                
                # If we reached the theoretical lower bound, we can stop
                if best_scope <= 112:
                    break
        else:
            # Reject, revert
            current_rows[r_idx][i_idx] = old_val
            
        T *= alpha
        if T < min_T:
            # Reheat to escape local minima
            T = 10.0
            
        steps += 1
        
        # Every 1000 steps, try a "Row Regeneration" move
        # This can help escape deep local minima where single element moves are insufficient
        if steps % 1000 == 0 and random.random() < 0.1:
            reg_idx = random.randint(0, n_rows - 1)
            # Compute diffs of other rows
            other_diffs = set()
            for i, r in enumerate(current_rows):
                if i == reg_idx: continue
                other_diffs.update(get_row_diffs(r))
            
            # Generate new row greedily, but with some randomness
            # We want to fill the row such that it doesn't conflict with other_diffs
            # and has a small max value.
            new_row = [0]
            x = 0
            success = True
            while len(new_row) < k + 1:
                x += 1
                # Try to add x
                new_diffs_row = {x - a for a in new_row}
                # Check against other_diffs and internal conflicts
                # Internal conflicts: if new_diffs_row has duplicates, it's invalid.
                # But since we build sequentially, new_diffs_row are all distinct?
                # No, e.g. row [0, 2, 4], adding 6: diffs 6,4,2. 4 and 2 are new.
                # Wait, the check `not (new_diffs_row & other_diffs)` only checks against OTHER rows.
                # We also need to ensure internal uniqueness.
                # Actually, if we just add x, the new diffs are x - a for a in row.
                # Are these distinct? Yes, because a are distinct.
                # Are they distinct from previous diffs in this row?
                # Previous diffs were (a_j - a_i) for j < len(row).
                # New diffs are (x - a_i).
                # Could x - a_i == a_j - a_k?
                # Yes. So we must check internal uniqueness too.
                
                internal_diffs = set()
                valid_internal = True
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        if d in internal_diffs:
                            valid_internal = False
                            break
                        internal_diffs.add(d)
                
                # Check if new diffs conflict with internal or other
                all_new_diffs = set(new_diffs_row)
                if all_new_diffs & internal_diffs:
                    # Conflict within the new row itself
                    # This means x - a_i == a_j - a_k
                    # This is a "Sidon" type constraint.
                    # We just skip this x.
                    continue
                
                if all_new_diffs & other_diffs:
                    # Conflict with other rows
                    continue
                
                # Check if new diffs conflict with each other?
                # They are x - a_i. Since a_i are distinct, these are distinct.
                
                # So if we passed the above checks, we can add x.
                # But wait, `internal_diffs` was computed BEFORE adding x.
                # So we need to check if any `x - a_i` is in `internal_diffs`.
                # We did that: `if all_new_diffs & internal_diffs`.
                
                # So if we reach here, x is safe to add.
                new_row.append(x)
                # Update internal_diffs for next iteration?
                # We can just recompute or update. Recomputing is O(n^2) but n is small (8).
                # Or we can maintain the set.
                # Let's just recompute in the loop for simplicity, or update.
                # Actually, the loop above recomputes internal_diffs from scratch every time.
                # That's O(n^3) total for the row. n=8, so 512 ops. Fine.
            
            if len(new_row) == k + 1:
                # Check if this new row is valid with the rest
                # We constructed it to be valid against other_diffs and internally.
                # So it should be valid.
                old_row = current_rows[reg_idx]
                current_rows[reg_idx] = new_row
                new_valid, new_scope, new_conflicts = is_valid(current_rows)
                new_cost = get_cost(current_rows)
                
                if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / T):
                    current_cost = new_cost
                    current_scope = new_scope
                    current_conflicts = new_conflicts
                    current_valid = new_valid
                    if current_valid and current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                        if best_scope <= 112:
                            break
                else:
                    current_rows[reg_idx] = old_row

    # Ensure we return a valid solution
    # If the best found is invalid, we might need to return the greedy one (which is valid)
    # But greedy is valid by construction.
    # So best_rows should always be valid if we started with a valid greedy solution
    # and only updated best when valid.
    
    return best_rows
