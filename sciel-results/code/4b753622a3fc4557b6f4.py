import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Each row has k+1 marks (indices 0..k), starting at 0
    # Total marks per row: k+1
    # Number of differences per row: k*(k+1)/2
    # Total differences across all rows: n_rows * k*(k+1)/2
    
    n_marks = k + 1  # marks per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Precompute all (row, i, j) pairs where i < j for differences
    # We represent a solution as a list of n_rows rows, each a list of n_marks integers
    # rows[r][0] = 0 always
    # rows[r][m] for m in 1..k are the variables
    
    # Greedy initialization
    def greedy_init():
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_marks:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Compute cost: maximum entry + penalty for conflicts
    def compute_cost(rows):
        # Check for conflicts and compute max entry
        used = set()
        max_entry = 0
        conflicts = 0
        for r in range(n_rows):
            row = rows[r]
            for m in range(n_marks):
                if row[m] > max_entry:
                    max_entry = row[m]
            # Check differences
            for i in range(n_marks):
                for j in range(i + 1, n_marks):
                    d = row[j] - row[i]
                    if d <= 0:
                        # Should not happen if strictly increasing, but handle
                        conflicts += 10000
                        continue
                    if d in used:
                        conflicts += 1
                    used.add(d)
        return max_entry + conflicts * 100
    
    # Simulated Annealing
    def simulated_annealing(rows, time_budget):
        current_rows = [list(r) for r in rows]
        current_cost = compute_cost(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost
        
        # Temperature schedule
        T_start = 100.0
        T_end = 0.01
        steps = 0
        max_steps = 500000
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_steps - 1))
        
        while time.time() < time_budget and steps < max_steps:
            steps += 1
            T = T_start * (cooling_rate ** steps)
            if T < T_end:
                T = T_end
            
            # Random move: pick a random row and a random mark position (1..k) and change it
            r = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # positions 1..k (position 0 is fixed at 0)
            
            # Ensure strict increase: new value must be > row[pos-1] and < row[pos+1] if pos < k
            old_val = current_rows[r][pos]
            
            # Determine valid range
            min_val = current_rows[r][pos - 1] + 1
            max_val = current_rows[r][pos + 1] - 1 if pos < k else 200  # upper bound
            
            if min_val > max_val:
                continue
            
            # Choose new value
            if random.random() < 0.5:
                # Small perturbation
                delta = random.randint(-5, 5)
                new_val = old_val + delta
            else:
                # Random in range
                new_val = random.randint(min_val, max_val)
            
            if new_val < min_val or new_val > max_val:
                continue
            
            if new_val == old_val:
                continue
            
            # Apply move
            current_rows[r][pos] = new_val
            new_cost = compute_cost(current_rows)
            
            delta_cost = new_cost - current_cost
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / max(T, 1e-10)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(row) for row in current_rows]
            else:
                # Revert
                current_rows[r][pos] = old_val
            
            # Occasionally try to reduce the maximum by shifting
            if steps % 1000 == 0:
                # Find the row with the largest entry and try to compress it
                max_r = max(range(n_rows), key=lambda r: current_rows[r][-1])
                max_val = current_rows[max_r][-1]
                if max_val > 0:
                    # Try to shift the entire row down by 1 if possible
                    # Check if all differences remain positive and unique
                    shifted = [0] + [v - 1 for v in current_rows[max_r][1:]]
                    if all(shifted[i] < shifted[i+1] for i in range(n_marks - 1)):
                        # Check conflicts
                        old_row = current_rows[max_r]
                        current_rows[max_r] = shifted
                        temp_cost = compute_cost(current_rows)
                        if temp_cost <= current_cost or random.random() < math.exp(-(temp_cost - current_cost) / max(T, 1e-10)):
                            current_cost = temp_cost
                            if temp_cost < best_cost:
                                best_cost = temp_cost
                                best_rows = [list(row) for row in current_rows]
                        else:
                            current_rows[max_r] = old_row
        
        return best_rows, best_cost
    
    # Local search: try to fix conflicts by adjusting individual marks
    def local_search(rows, time_budget):
        current_rows = [list(r) for r in rows]
        current_cost = compute_cost(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_cost = current_cost
        
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < time_budget and no_improve < max_no_improve:
            improved = False
            # Try each row and each position
            for r in range(n_rows):
                for pos in range(1, k + 1):
                    min_val = current_rows[r][pos - 1] + 1
                    max_val = current_rows[r][pos + 1] - 1 if pos < k else 200
                    
                    if min_val > max_val:
                        continue
                    
                    best_local_val = current_rows[r][pos]
                    best_local_cost = current_cost
                    
                    # Try a few candidate values
                    candidates = set()
                    candidates.add(current_rows[r][pos])
                    candidates.add(min_val)
                    candidates.add(max_val)
                    if current_rows[r][pos] > min_val:
                        candidates.add(current_rows[r][pos] - 1)
                    if current_rows[r][pos] < max_val:
                        candidates.add(current_rows[r][pos] + 1)
                    # Add some random candidates
                    for _ in range(3):
                        candidates.add(random.randint(min_val, max_val))
                    
                    for val in candidates:
                        if val < min_val or val > max_val:
                            continue
                        current_rows[r][pos] = val
                        new_cost = compute_cost(current_rows)
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local_val = val
                    
                    current_rows[r][pos] = best_local_val
                    if best_local_cost < current_cost:
                        current_cost = best_local_cost
                        improved = True
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [list(row) for row in current_rows]
        
        return best_rows, best_cost
    
    # Get initial solution
    greedy_rows = greedy_init()
    
    # Run simulated annealing
    sa_rows, sa_cost = simulated_annealing(greedy_rows, time.time() + 40)
    
    # Run local search from the SA result
    ls_rows, ls_cost = local_search(sa_rows, time.time() + 80)
    
    # Try a few random restarts if time permits
    best_rows = ls_rows
    best_cost = ls_cost
    
    remaining = deadline - time.time()
    if remaining > 10:
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            perturbed = [list(r) for r in best_rows]
            for _ in range(5):
                r = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                min_val = perturbed[r][pos - 1] + 1
                max_val = perturbed[r][pos + 1] - 1 if pos < k else 200
                if min_val <= max_val:
                    perturbed[r][pos] = random.randint(min_val, max_val)
            
            restart_budget = deadline - 5 - restart * 10
            sa2_rows, sa2_cost = simulated_annealing(perturbed, restart_budget)
            if sa2_cost < best_cost:
                best_cost = sa2_cost
                best_rows = sa2_rows
    
    # Final validation and cleanup
    # Ensure strictly increasing
    for r in range(n_rows):
        for m in range(1, n_marks):
            if best_rows[r][m] <= best_rows[r][m - 1]:
                # Force strictly increasing
                best_rows[r][m] = best_rows[r][m - 1] + 1
    
    return best_rows
