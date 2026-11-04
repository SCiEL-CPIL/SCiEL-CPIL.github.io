"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    best_rows = None
    best_max = float('inf')
    
    def get_score(rows):
        """
        Calculate the objective function for a given configuration.
        We want to minimize the largest entry, subject to validity.
        To handle validity in SA, we use a penalty for conflicts.
        
        Score = max_entry + penalty * number_of_conflicts
        If there are no conflicts, this is just max_entry.
        """
        if not rows:
            return float('inf')
            
        max_entry = 0
        used_diffs = set()
        conflicts = 0
        
        for r in range(n_rows):
            row = rows[r]
            # Check strictly increasing and starting at 0
            if row[0] != 0:
                conflicts += 1000 # Heavy penalty
            for i in range(len(row)):
                if row[i] > max_entry:
                    max_entry = row[i]
                if i > 0 and row[i] <= row[i-1]:
                    conflicts += 1000 # Heavy penalty for non-increasing
            
            # Check differences
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        conflicts += 1
                    else:
                        used_diffs.add(d)
                        
        # Penalty needs to be large enough to discourage conflicts 
        # but small enough to allow exploration if necessary.
        # However, for this specific task, we want to find a valid solution first.
        # A common strategy is: if valid, return max_entry. If invalid, return max_entry + large_penalty.
        # To make SA work, we can define:
        # cost = max_entry if valid else max_entry + 100 * conflicts
        
        # Actually, a simpler cost for SA that drives towards validity and small max:
        # cost = max_entry + conflicts * (max_entry + 1)
        # This ensures any valid solution is better than any invalid one with similar max.
        
        return max_entry + conflicts * (max_entry + 1)

    def generate_initial_rows(seed):
        """Generate a random valid-ish starting point using greedy with randomization."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Random step size to avoid immediate collisions
                step = rng.randint(1, 5)
                x += step
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                else:
                    # If collision, try a larger jump
                    x += rng.randint(5, 15)
            rows.append(row)
        return rows

    def simulated_annealing(seed):
        """
        Run Simulated Annealing to improve a set of rows.
        Returns the best rows found during the run.
        """
        nonlocal best_rows, best_max
        
        rng = random.Random(seed)
        
        # Start with a greedy solution
        current_rows = generate_initial_rows(seed)
        current_cost = get_score(current_rows)
        
        best_local_rows = [row[:] for row in current_rows]
        best_local_cost = current_cost
        
        if current_cost < best_max:
            best_max = current_cost
            best_rows = [row[:] for row in current_rows]
            
        # Annealing parameters
        T = 1000.0  # Initial temperature
        T_min = 1.0
        alpha = 0.995  # Cooling rate
        steps_per_temp = 100
        
        while T > T_min and time.time() < deadline:
            for _ in range(steps_per_temp):
                if time.time() > deadline:
                    return best_local_rows
                
                # Pick a random move
                # 1. Pick a random row
                r = rng.randint(0, n_rows - 1)
                # 2. Pick a random column (1 to k, since 0 is fixed)
                c = rng.randint(1, k)
                
                old_val = current_rows[r][c]
                
                # Generate a new value
                # We want to explore around the current value, but also allow larger jumps
                # to escape local minima.
                # Constraint: must be > previous element, < next element (if exists)
                min_val = current_rows[r][c-1] + 1
                max_val = current_rows[r][c+1] - 1 if c < k else 10000 # Upper bound heuristic
                
                if min_val > max_val:
                    # Invalid state structure, try to fix by resetting this part? 
                    # Or just skip this move.
                    continue
                
                # Randomize the new value
                # Sometimes small change, sometimes big change
                if rng.random() < 0.5:
                    # Small local move
                    delta = rng.randint(-10, 10)
                    new_val = old_val + delta
                else:
                    # Larger jump
                    new_val = rng.randint(min_val, max_val)
                
                # Ensure strictly increasing
                if new_val <= current_rows[r][c-1]:
                    new_val = current_rows[r][c-1] + 1
                if c < k and new_val >= current_rows[r][c+1]:
                    new_val = current_rows[r][c+1] - 1
                
                if new_val <= current_rows[r][c-1] or (c < k and new_val >= current_rows[r][c+1]):
                    continue
                    
                # Apply move
                current_rows[r][c] = new_val
                new_cost = get_score(current_rows)
                
                # Acceptance criterion
                if new_cost < current_cost or rng.random() < math.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_rows = [row[:] for row in current_rows]
                        if new_cost < best_max:
                            best_max = new_cost
                            best_rows = [row[:] for row in current_rows]
                else:
                    # Revert
                    current_rows[r][c] = old_val
            
            T *= alpha
            
        return best_local_rows

    # Run multiple SA restarts
    seed = 0
    while time.time() < deadline:
        seed += 1
        simulated_annealing(seed)
        
        # If we hit the lower bound, we can't do better
        if best_max == lower_bound:
            break

    # Fallback if SA failed completely (shouldn't happen, but just in case)
    if best_rows is None:
        # Simple greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)

    return best_rows
# EVOLVE-BLOCK-END
