import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit (or 150s if that's the real one, but 118 is safe for 120)
    
    # Total number of differences that must be distinct
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope: M(M+1)/2 >= total_diffs
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    def get_scope_from_diffs(diff_rows):
        """Calculate the maximum element (scope) from difference rows."""
        max_val = 0
        for d_row in diff_rows:
            current_sum = 0
            for d in d_row:
                current_sum += d
            if current_sum > max_val:
                max_val = current_sum
        return max_val

    def get_diffs_set(diff_rows):
        """Calculate the set of all pairwise differences from difference rows."""
        used = set()
        for d_row in diff_rows:
            # Reconstruct the row values to get pairwise diffs?
            # Or calculate directly?
            # Row values: v[0]=0, v[i] = sum(d_row[0..i-1])
            # Pairwise diff v[j]-v[i] = sum(d_row[i..j-1])
            # It's easier to just reconstruct the row.
            row_vals = [0]
            curr = 0
            for d in d_row:
                curr += d
                row_vals.append(curr)
            
            for i in range(len(row_vals)):
                for j in range(i + 1, len(row_vals)):
                    d = row_vals[j] - row_vals[i]
                    used.add(d)
        return used

    def count_collisions(diff_rows):
        """Return the number of missing differences (collisions)."""
        # We need total_diffs unique values.
        # If we have a set of size S, collisions = total_diffs - S?
        # No, if there are duplicates, the set size is smaller.
        # The number of "colliding" pairs is total_diffs - len(set).
        # But we just need to know if it's 0.
        # For energy, we want a penalty proportional to the number of duplicates.
        # Let's just count how many differences are NOT unique.
        # Actually, simpler: 
        # Total possible pairs = total_diffs.
        # If set size is S, then number of "lost" distinct values is total_diffs - S.
        # This is a good penalty.
        s = get_diffs_set(diff_rows)
        return total_diffs - len(s)

    def greedy_initial():
        """
        Generate an initial valid solution using greedy on values.
        Returns list of rows (values).
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def convert_to_diffs(rows):
        """Convert value rows to difference rows."""
        diff_rows = []
        for row in rows:
            d_row = []
            for i in range(1, len(row)):
                d_row.append(row[i] - row[i-1])
            diff_rows.append(d_row)
        return diff_rows

    def convert_to_values(diff_rows):
        """Convert difference rows to value rows."""
        val_rows = []
        for d_row in diff_rows:
            row = [0]
            curr = 0
            for d in d_row:
                curr += d
                row.append(curr)
            val_rows.append(row)
        return val_rows

    # Phase 1: Initial Greedy
    init_rows = greedy_initial()
    init_diffs = convert_to_diffs(init_rows)
    
    best_rows = init_rows
    best_scope = get_scope_from_diffs(init_diffs)
    
    # Phase 2: Simulated Annealing on Difference Vectors
    current_diffs = [r[:] for r in init_diffs]
    current_scope = get_scope_from_diffs(current_diffs)
    current_collisions = count_collisions(current_diffs)
    
    # Energy function
    # We want to minimize scope, but must have 0 collisions.
    # Penalty for collisions should be very high.
    PENALTY = 1000
    
    def energy(diff_rows):
        scope = get_scope_from_diffs(diff_rows)
        coll = count_collisions(diff_rows)
        return scope + PENALTY * coll

    current_energy = energy(current_diffs)
    
    # SA Parameters
    # Start with a temperature that allows some scope increases
    # Scope is around 100-200. Delta E can be small.
    T = 10.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    steps = 0
    no_improve_steps = 0
    
    while time.time() < deadline:
        # Pick a random row and a random index in that row's difference vector
        r_idx = random.randint(0, n_rows - 1)
        d_idx = random.randint(0, k - 1)
        
        # Perturbation: change the difference by a random amount
        # Smaller changes are more likely to be valid if we are close to optimal
        # Larger changes help escape deep local minima
        if random.random() < 0.8:
            delta = random.choice([-2, -1, 1, 2])
        else:
            delta = random.randint(-5, 5)
            if delta == 0:
                delta = 1
            
        new_diff_val = current_diffs[r_idx][d_idx] + delta
        
        # Constraint: differences must be positive integers
        if new_diff_val <= 0:
            # Reject
            # Optionally, try a smaller step or just skip
            continue
            
        # Apply move
        old_diff_val = current_diffs[r_idx][d_idx]
        current_diffs[r_idx][d_idx] = new_diff_val
        
        # Calculate new energy
        new_scope = get_scope_from_diffs(current_diffs)
        new_collisions = count_collisions(current_diffs)
        new_energy = new_scope + PENALTY * new_collisions
        
        delta_E = new_energy - current_energy
        
        # Acceptance criterion
        if delta_E < 0 or random.random() < math.exp(-delta_E / max(T, 0.001)):
            # Accept
            current_scope = new_scope
            current_collisions = new_collisions
            current_energy = new_energy
            
            if current_collisions == 0:
                if current_scope < best_scope:
                    best_scope = current_scope
                    # Convert to values for storage
                    best_rows = convert_to_values(current_diffs)
                    no_improve_steps = 0
                    if best_scope <= lower_bound:
                        return best_rows
                else:
                    no_improve_steps += 1
            else:
                # If we are invalid, we don't update best_rows, but we keep searching
                # to try to get back to a valid state
                no_improve_steps += 1
        else:
            # Reject: revert
            current_diffs[r_idx][d_idx] = old_diff_val
        
        # Cool down
        T *= cooling_rate
        
        # Occasionally, if stuck in invalid state for too long, restart from best valid?
        # Or just keep going. The penalty should guide it back.
        
        # If T is too low, restart SA?
        if T < min_T:
            # Restart with a new T if we haven't found the lower bound
            # Or just break if we are stuck?
            # Let's restart T to continue exploring
            T = 5.0
            # Also, maybe randomize the current state slightly to escape?
            # If we are in a valid state, we can perturb it.
            # If we are in an invalid state, we are already perturbed.
            # Let's just keep going.
            
        steps += 1
        
        # Safety check: if no improvement for a long time and T is low, 
        # maybe restart from a fresh greedy?
        if no_improve_steps > 5000 and T < 1.0:
            # Generate a new random initial solution?
            # Or just restart T?
            T = 10.0
            no_improve_steps = 0
            # Optional: perturb current solution significantly
            for _ in range(10):
                r_idx = random.randint(0, n_rows - 1)
                d_idx = random.randint(0, k - 1)
                current_diffs[r_idx][d_idx] = max(1, current_diffs[r_idx][d_idx] + random.randint(-5, 5))
            current_energy = energy(current_diffs)
            current_scope = get_scope_from_diffs(current_diffs)
            current_collisions = count_collisions(current_diffs)

    # Final check: ensure best_rows is valid
    # It should be, as we only update it when collisions == 0
    return best_rows
