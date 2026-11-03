"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    
    # Problem Parameters
    num_vars = n_rows * (k + 1)
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row

    # Helper: Calculate all differences for a single row
    def get_row_diffs(row):
        diffs = []
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                diffs.append(row[j] - row[i])
        return diffs

    # Helper: Calculate all differences for the entire set of rows
    def get_all_diffs(rows):
        diffs = []
        for r in rows:
            diffs.extend(get_row_diffs(r))
        return diffs

    # Helper: Check validity and get score
    # Returns (is_valid, score)
    def evaluate(rows):
        # Check row validity (strictly increasing)
        for r in rows:
            if r[0] != 0:
                return False, float('inf')
            for i in range(1, k + 1):
                if r[i] <= r[i-1]:
                    return False, float('inf')
        
        # Check difference uniqueness
        diffs = get_all_diffs(rows)
        if len(set(diffs)) != len(diffs):
            return False, float('inf')
        
        score = max(max(r) for r in rows)
        return True, score

    # Helper: Fitness for Simulated Annealing (lower is better)
    # Penalty for collisions + penalty for scope
    def fitness(rows):
        # Check row validity first
        for r in rows:
            if r[0] != 0:
                return 1e10
            for i in range(1, k + 1):
                if r[i] <= r[i-1]:
                    return 1e10
        
        diffs = get_all_diffs(rows)
        unique_diffs = set(diffs)
        collisions = len(diffs) - len(unique_diffs)
        
        scope = max(max(r) for r in rows)
        
        # Weighting: collisions are fatal, scope is the goal
        # If valid, fitness is just scope.
        # If invalid, fitness is huge + collisions.
        if collisions == 0:
            return scope
        else:
            return 10000 + collisions * 100 + scope

    # 1. Initialization: Generate a greedy solution
    # Strategy: Greedy construction with backtracking for diversity? 
    # For SA, a good start is crucial. Let's use a greedy solution.
    def greedy_init():
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

    current_rows = greedy_init()
    current_fitness = fitness(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_fitness = current_fitness

    # 2. Simulated Annealing Loop
    # Parameters
    T = 1000.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.1
    
    # We need a fast way to update fitness when one value changes.
    # However, for simplicity and given the small size, recalculating is fine initially.
    # But let's try to be slightly smarter if possible. 
    # Actually, recalculating all diffs for n_rows=4, k=8 is very fast.
    # 4 * 36 = 144 diffs. Set operations are fast.
    
    while time.time() < deadline and T > min_T:
        # Generate a neighbor
        neighbor_rows = [r[:] for r in current_rows]
        
        # Pick a random row and a random position (1..k)
        r_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        # Random walk step
        # Bias towards smaller steps later?
        step_size = random.randint(-10, 10)
        if random.random() < 0.5:
             step_size = random.randint(-2, 2) # More fine-grained moves
        
        old_val = neighbor_rows[r_idx][pos]
        new_val = old_val + step_size
        
        # Ensure strict increase
        if pos > 0:
            if new_val <= neighbor_rows[r_idx][pos-1]:
                new_val = neighbor_rows[r_idx][pos-1] + 1
        if pos < k:
            if new_val >= neighbor_rows[r_idx][pos+1]:
                new_val = neighbor_rows[r_idx][pos+1] - 1
        
        # Ensure positive
        if new_val < 1:
            new_val = 1
            
        # If clamping didn't change it, skip
        if new_val == old_val:
            T *= cooling_rate
            continue
            
        neighbor_rows[r_idx][pos] = new_val
        
        # Repair subsequent values if necessary to maintain strict increase
        # If we increased a value, subsequent ones might be too small
        for j in range(pos + 1, k + 1):
            if neighbor_rows[r_idx][j] <= neighbor_rows[r_idx][j-1]:
                neighbor_rows[r_idx][j] = neighbor_rows[r_idx][j-1] + 1
        # Backward pass repair
        for j in range(pos - 1, 0, -1):
            if neighbor_rows[r_idx][j] >= neighbor_rows[r_idx][j+1]:
                neighbor_rows[r_idx][j] = neighbor_rows[r_idx][j+1] - 1
                if neighbor_rows[r_idx][j] < 1:
                    # If we can't maintain validity, revert this row or reject move
                    # Let's just reject the move if it becomes invalid
                    neighbor_rows = [r[:] for r in current_rows]
                    T *= cooling_rate
                    break
        
        if neighbor_rows == current_rows:
             T *= cooling_rate
             continue

        neighbor_fitness = fitness(neighbor_rows)
        
        # Acceptance criterion
        delta_f = neighbor_fitness - current_fitness
        
        if delta_f < 0:
            # Better solution
            current_rows = neighbor_rows
            current_fitness = neighbor_fitness
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_rows = [r[:] for r in current_rows]
        else:
            # Worse solution, accept with probability exp(-delta_f / T)
            # If fitness is huge (invalid), delta_f is huge, prob is ~0
            if T > 0:
                p = math.exp(-delta_f / T)
                if random.random() < p:
                    current_rows = neighbor_rows
                    current_fitness = neighbor_fitness
        
        T *= cooling_rate
        
        # Optional: Restart if stuck for too long? 
        # Simple restart: if no improvement for X iterations, reset to best and reheat
        # For now, simple cooling.

    # Final check: return the best valid solution
    is_val, score = evaluate(best_rows)
    if not is_val:
        # Fallback to greedy
        best_rows = greedy_init()
        
    return best_rows
# EVOLVE-BLOCK-END
