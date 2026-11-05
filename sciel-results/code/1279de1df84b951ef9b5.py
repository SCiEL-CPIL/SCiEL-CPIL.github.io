# EVOLVE-BLOCK-START
import time
import random
import math
import sys

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s limit

    # 1. Greedy Baseline
    def greedy_construct(n_rows, k, seed=0):
        used = set()
        rows = []
        rng = random.Random(seed)
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

    best_rows = greedy_construct(n_rows, k)
    best_max = max(max(row) for row in best_rows)
    
    # 2. Simulated Annealing
    # State: list of rows.
    # Energy: 
    #   - Collision penalty: number of duplicate differences * weight
    #   - Scope penalty: max_entry^2 * weight
    
    # Precompute differences for a row
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def compute_energy(rows, max_entry_cap=None):
        # Count differences
        # For efficiency, we can use a dictionary or Counter
        # But for small k (like 8), the number of diffs per row is k*(k+1)/2 = 36.
        # Total diffs = n_rows * 36.
        # n_rows is small (4). Total diffs ~ 144.
        
        # We need to count collisions.
        # A difference is a collision if it appears more than once.
        # Energy component 1: sum over all diffs d of (count[d] - 1) if count[d] > 1
        # This is equivalent to total_diffs - number_of_unique_diffs.
        
        # Energy component 2: max_entry^2
        # We want to minimize max entry.
        
        # Let's calculate unique diffs count
        all_diffs = []
        max_entry = 0
        for row in rows:
            if row[-1] > max_entry:
                max_entry = row[-1]
            all_diffs.extend(get_diffs(row))
        
        # Count uniques
        # Using a set is O(N)
        unique_diffs = set(all_diffs)
        collisions = len(all_diffs) - len(unique_diffs)
        
        # Penalty for exceeding max_entry_cap if provided? 
        # The SA should explore. The max_entry is determined by the state.
        # We want to minimize max_entry.
        # So energy = collisions * W_coll + max_entry^2 * W_scope
        
        return collisions, max_entry

    # Initialize SA state from greedy
    current_rows = [row[:] for row in best_rows]
    
    # We will track the best valid solution found so far
    # A solution is valid if collisions == 0.
    # Among valid solutions, we want the one with min max_entry.
    
    best_valid_rows = best_rows
    best_valid_max = best_max
    
    # SA Parameters
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    alpha = 0.995 # cooling rate
    
    # Weights
    W_coll = 10.0
    W_scope = 0.1
    
    def get_energy_val(rows):
        coll, m = compute_energy(rows)
        return coll * W_coll + m * m * W_scope

    current_energy = get_energy_val(current_rows)
    
    T = T_start
    
    # Move generators
    # 1. Increment/Decrement an entry (maintaining strictly increasing order)
    # 2. Swap two entries? No, must remain increasing.
    # 3. Shift a segment?
    
    # Let's stick to simple moves:
    # Pick a random row r, random index i (1..k), random delta +/- 1, +/- 2
    # Ensure row[r][i] + delta > row[r][i-1] and row[r][i] + delta < row[r][i+1] (if exists)
    # Also row[r][i] >= 0
    
    rng = random.Random(42)
    
    steps = 0
    max_steps = 500000
    
    while T > T_end and steps < max_steps:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Generate move
        r = rng.randint(0, n_rows - 1)
        i = rng.randint(1, k) # indices 1 to k
        
        # Determine valid delta range
        # row[r][i] must be > row[r][i-1] and < row[r][i+1] (if i < k)
        min_val = current_rows[r][i-1] + 1
        max_val = current_rows[r][i+1] - 1 if i < k else 10000 # Arbitrary large upper bound
        
        # Try a small random change
        # Bias towards smaller values to reduce scope
        # delta in [-3, 3]
        delta = rng.randint(-3, 3)
        if delta == 0:
            delta = 1 if rng.random() < 0.5 else -1
            
        new_val = current_rows[r][i] + delta
        
        if new_val < min_val or new_val > max_val:
            # Try to clamp or skip
            # Clamping might be good for scope reduction
            if new_val < min_val:
                new_val = min_val
            elif new_val > max_val:
                new_val = max_val
                
            if new_val == current_rows[r][i]:
                T *= alpha
                steps += 1
                continue

        # Apply move
        old_val = current_rows[r][i]
        current_rows[r][i] = new_val
        
        # Calculate new energy
        # To speed up, we could incremental update, but for k=8, n_rows=4,
        # recomputing is fast enough? 
        # 144 diffs. Set creation is fast.
        new_energy = get_energy_val(current_rows)
        
        # Acceptance
        dE = new_energy - current_energy
        if dE < 0 or rng.random() < math.exp(-dE / T):
            current_energy = new_energy
            steps += 1
            
            # Check if valid and better
            coll, m = compute_energy(current_rows)
            if coll == 0:
                if m < best_valid_max:
                    best_valid_max = m
                    best_valid_rows = [row[:] for row in current_rows]
        else:
            # Revert
            current_rows[r][i] = old_val
        
        T *= alpha
        
    # Final local search / polish on best_valid_rows
    # Try to reduce max entry by local moves that keep validity
    # We can run a short SA again starting from best_valid_rows with lower T
    
    # Re-initialize for polish
    current_rows = [row[:] for row in best_valid_rows]
    current_energy = get_energy_val(current_rows)
    T = 5.0
    polish_steps = 50000
    
    while T > 0.1 and polish_steps > 0:
        if time.time() - start_time > time_limit - 2:
            break
            
        r = rng.randint(0, n_rows - 1)
        i = rng.randint(1, k)
        
        min_val = current_rows[r][i-1] + 1
        max_val = current_rows[r][i+1] - 1 if i < k else 10000
        
        delta = rng.randint(-2, 2)
        if delta == 0:
            delta = 1 if rng.random() < 0.5 else -1
            
        new_val = current_rows[r][i] + delta
        
        if new_val < min_val or new_val > max_val:
            if new_val < min_val:
                new_val = min_val
            elif new_val > max_val:
                new_val = max_val
            if new_val == current_rows[r][i]:
                T *= 0.98
                polish_steps -= 1
                continue

        old_val = current_rows[r][i]
        current_rows[r][i] = new_val
        
        new_energy = get_energy_val(current_rows)
        
        dE = new_energy - current_energy
        if dE < 0 or rng.random() < math.exp(-dE / T):
            current_energy = new_energy
            coll, m = compute_energy(current_rows)
            if coll == 0 and m < best_valid_max:
                best_valid_max = m
                best_valid_rows = [row[:] for row in current_rows]
        else:
            current_rows[r][i] = old_val
            
        T *= 0.98
        polish_steps -= 1
        
    # Return the best valid solution found
    # If no valid solution was found (should not happen given greedy init), return greedy
    # best_valid_rows is always valid by construction (initialized with greedy)
    
    return best_valid_rows
# EVOLVE-BLOCK-END
