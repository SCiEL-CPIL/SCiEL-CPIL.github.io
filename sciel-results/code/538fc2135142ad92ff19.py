import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    # 1. Greedy Solution to get an initial upper bound and a valid starting point
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
        
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in rows)
    
    # Helper to calculate the number of conflicts (duplicate differences)
    def get_conflicts(current_rows):
        # Count frequencies of all differences
        freq = {}
        for row in current_rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    freq[d] = freq.get(d, 0) + 1
        
        # A conflict exists for every difference that appears more than once.
        # The number of "excess" occurrences is sum(count - 1) for count > 1.
        # Alternatively, total_diffs - unique_diffs is the number of duplicates.
        # Let's use sum(count - 1) for count > 1. This is the number of elements 
        # that would need to be removed to make all differences distinct.
        conflicts = 0
        for count in freq.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    # Initial state: We want to minimize scope, but we must satisfy the distinctness constraint.
    # We start with the greedy solution (valid, but high scope).
    # We will try to reduce the scope by shrinking the rows, but we allow conflicts during the search.
    # Objective: Minimize (conflicts * LARGE_PENALTY + scope)
    # If conflicts > 0, the solution is invalid. We want to drive conflicts to 0 while minimizing scope.
    
    # To make SA effective, we should start from a valid solution and try to "squeeze" it.
    # However, standard SA works on a fixed set of variables. Here, the "variables" are the positions.
    # A better approach for this specific constraint:
    # Start with the greedy solution.
    # Randomly pick a row and a position, and try to decrease the value (squeeze).
    # If it creates a conflict, we might still accept it if we are in a "high temperature" phase, 
    # but we need a way to get back to validity.
    
    # Alternative: Start with a "packed" but invalid solution?
    # No, let's stick to the "shrink" strategy.
    # State: list of rows.
    # Move: Pick a random row, pick a random index i (1..k), decrease row[i] by a random amount 
    # such that it remains > row[i-1].
    # If this move reduces the total scope, we accept it.
    # If it increases conflicts, we accept it with probability exp(-delta_E / T).
    # If it reduces conflicts, we accept it (or with high prob).
    
    # Actually, a more robust approach for "minimize max value subject to constraints" is:
    # 1. Start with a valid solution (greedy).
    # 2. Try to reduce the largest element.
    # 3. If reducing it causes conflicts, we have to "fix" the conflicts by moving other elements.
    
    # Let's use a standard SA on the "conflict count" and "scope".
    # But we must ensure we don't get stuck in an invalid state forever.
    # Strategy: 
    #   - Start with Greedy (Valid, Scope S0).
    #   - Try to reduce scope by moving elements down.
    #   - If a move creates conflicts, accept with probability based on temperature.
    #   - If a move resolves conflicts, always accept (or high prob).
    #   - Periodically, if we are stuck in high conflicts, we might need to "expand" (increase values) 
    #     to resolve conflicts, but that increases scope. 
    #     Actually, if we are trying to minimize scope, we should only move elements DOWN or stay.
    #     But moving down might create new conflicts. 
    #     If we can't move down without creating conflicts, we are stuck at a local minimum of scope 
    #     for the current configuration. 
    #     To escape, we might need to temporarily increase some values (increase scope) to allow 
    #     others to decrease more significantly later? This is complex.
    
    # Simpler Robust Strategy:
    # 1. Generate a random valid solution? No, hard.
    # 2. Start with Greedy.
    # 3. Use SA to minimize Scope, but with a heavy penalty for Conflicts.
    #    Energy = Scope + Lambda * Conflicts.
    #    Lambda is large.
    #    Moves: Pick a random element (row, col) and change its value to a random valid value 
    #    (maintaining increasing order within the row).
    #    This allows "expanding" and "contracting".
    
    # Let's refine the move set.
    # If we pick row r and index i, we can change rows[r][i] to any value x such that 
    # rows[r][i-1] < x < rows[r][i+1] (if i < k).
    # If i == k, x can be anything > rows[r][k-1].
    
    # To bias towards smaller values, we can sample x from a distribution that favors smaller numbers,
    # or simply try to decrease the value first.
    
    # Let's try this:
    # 1. Start with Greedy.
    # 2. T = 1.0 (initial temperature).
    # 3. Loop until deadline:
    #    - Pick random row r, random index i (1..k).
    #    - Generate a new value x for rows[r][i].
    #      - Bias: Try to decrease. x = rows[r][i] - random_int(1, 5)
    #      - Ensure x > rows[r][i-1].
    #      - If i < k, ensure x < rows[r][i+1].
    #    - Calculate Delta_E = (New_Scope + Lambda * New_Conflicts) - (Old_Scope + Lambda * Old_Conflicts)
    #    - If Delta_E < 0, accept.
    #    - Else, accept with prob exp(-Delta_E / T).
    #    - Cool T.
    
    # This might get stuck if it can't reduce conflicts. 
    # If we start valid, and we only move down, we might create conflicts and never resolve them 
    # because resolving them often requires moving UP (increasing scope).
    # So we MUST allow moves that increase scope (move up) to resolve conflicts.
    
    # Modified Move Generation:
    # - 50% chance: Try to decrease (shrink).
    # - 50% chance: Try to increase (expand) or random shift.
    # This allows the system to "breathe" and find better packings.
    
    # Let's implement this.
    
    # Parameters
    LAMBDA = 1000  # Penalty for conflicts. Must be larger than max possible scope difference.
    # Max scope is around 134. So 1000 is safe.
    
    current_rows = [r[:] for r in rows]
    current_scope = best_scope
    current_conflicts = 0 # Greedy is valid
    
    # Precompute row lengths
    row_len = k + 1
    
    def calc_scope_conflicts(rws):
        scope = max(max(r) for r in rws)
        conflicts = get_conflicts(rws)
        return scope, conflicts
    
    # Initial check
    current_scope, current_conflicts = calc_scope_conflicts(current_rows)
    
    best_scope_found = current_scope
    best_rows_found = [r[:] for r in current_rows]
    
    # SA Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    
    # We need to be careful with the time limit.
    # Each step involves calculating conflicts, which is O(n_rows * k^2).
    # n_rows=4, k=7 -> 4 * 28 = 112 diffs. Very fast.
    # We can do many steps.
    
    while time.time() < deadline:
        if T < min_T:
            # Reheat if stuck? Or just restart?
            # Let's try a quick reheat if no improvement for a long time.
            # For simplicity, just stop if T is too low, or reset.
            # Let's reset to a slightly perturbed version of the best found so far.
            current_rows = [r[:] for r in best_rows_found]
            current_scope, current_conflicts = calc_scope_conflicts(current_rows)
            T = 0.5 * T  # Reheat to half of original? Or just 1.0?
            T = 1.0
            # Perturb
            for _ in range(10):
                r = random.randint(0, n_rows - 1)
                i = random.randint(1, k)
                # Small random change
                delta = random.randint(-5, 5)
                new_val = current_rows[r][i] + delta
                # Check bounds
                if i > 0 and new_val <= current_rows[r][i-1]:
                    new_val = current_rows[r][i-1] + 1
                if i < k and new_val >= current_rows[r][i+1]:
                    new_val = current_rows[r][i+1] - 1
                if new_val < 0:
                    new_val = 0
                current_rows[r][i] = new_val
            current_scope, current_conflicts = calc_scope_conflicts(current_rows)
        
        # Generate Move
        r = random.randint(0, n_rows - 1)
        i = random.randint(1, k) # Don't change 0
        
        old_val = current_rows[r][i]
        
        # Decide direction
        if random.random() < 0.5:
            # Try to decrease
            # Max decrease is limited by previous element
            min_val = current_rows[r][i-1] + 1
            if old_val > min_val:
                # Decrease by random amount
                max_dec = old_val - min_val
                dec = random.randint(1, max(1, max_dec // 2)) # Bias towards small changes? Or large?
                # Let's try a random step size
                dec = random.randint(1, max(1, max_dec))
                new_val = old_val - dec
            else:
                new_val = old_val
        else:
            # Try to increase or random
            # Increase by random amount
            max_inc = 10 # Cap increase to avoid huge jumps
            inc = random.randint(1, max_inc)
            new_val = old_val + inc
            # Check upper bound if not last element
            if i < k:
                max_val = current_rows[r][i+1] - 1
                if new_val > max_val:
                    new_val = max_val
            if new_val < old_val:
                new_val = old_val # Don't decrease in "increase" mode
        
        if new_val == old_val:
            T *= cooling_rate
            continue
            
        # Apply move
        current_rows[r][i] = new_val
        
        new_scope, new_conflicts = calc_scope_conflicts(current_rows)
        
        old_energy = current_scope + LAMBDA * current_conflicts
        new_energy = new_scope + LAMBDA * new_conflicts
        
        delta_e = new_energy - old_energy
        
        if delta_e < 0:
            # Accept
            current_scope = new_scope
            current_conflicts = new_conflicts
            
            # Update best if valid
            if current_conflicts == 0 and current_scope < best_scope_found:
                best_scope_found = current_scope
                best_rows_found = [row[:] for row in current_rows]
        else:
            # Accept with probability
            if delta_e == 0:
                # Accept
                current_scope = new_scope
                current_conflicts = new_conflicts
                if current_conflicts == 0 and current_scope < best_scope_found:
                    best_scope_found = current_scope
                    best_rows_found = [row[:] for row in current_rows]
            else:
                prob = math.exp(-delta_e / T)
                if random.random() < prob:
                    current_scope = new_scope
                    current_conflicts = new_conflicts
                    if current_conflicts == 0 and current_scope < best_scope_found:
                        best_scope_found = current_scope
                        best_rows_found = [row[:] for row in current_rows]
                else:
                    # Reject: revert
                    current_rows[r][i] = old_val
                    # Scope and conflicts remain as they were (we didn't update them)
                    # Note: current_scope and current_conflicts variables still hold the OLD values
                    # because we only updated them in the accept branches.
                    # Wait, if we reject, we revert the row, and the scope/conflicts variables
                    # should still reflect the state before the move.
                    # In the code above, I only updated current_scope/conflicts on accept.
                    # So on reject, they are already correct (old values).
                    # And the row is reverted.
                    pass
        
        T *= cooling_rate
        
    # Return the best valid solution found
    # If best_rows_found is valid, return it.
    # If not (shouldn't happen if we started valid and only updated on valid), return best_rows_found.
    # But what if we never found a valid one better than greedy? best_rows_found is greedy.
    # What if the greedy solution was the only valid one?
    # We must ensure we return a valid solution.
    # The greedy solution is always valid.
    # If best_scope_found is from a valid state, it's fine.
    
    # Sanity check: Ensure the returned solution is valid.
    # If for some reason best_rows_found is invalid (bug), fall back to greedy.
    # But the logic above only updates best_rows_found if current_conflicts == 0.
    # And initial best_rows_found is greedy (valid).
    # So best_rows_found should always be valid.
    
    return best_rows_found
