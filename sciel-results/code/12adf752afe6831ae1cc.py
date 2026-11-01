import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Generate initial greedy solution
    def generate_greedy():
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

    best_rows = generate_greedy()
    best_cost = float('inf')
    
    # Helper to calculate cost
    def calculate_cost(rows):
        used = set()
        max_val = 0
        conflicts = 0
        for row in rows:
            max_val = max(max_val, max(row))
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
        # We want valid solutions (conflicts == 0) with min max_val
        # Cost = conflicts * 1000 + max_val
        return conflicts * 1000 + max_val, max_val, conflicts

    cur_cost, cur_max, cur_conf = calculate_cost(best_rows)
    best_rows_copy = [r[:] for r in best_rows]
    best_cost = cur_cost

    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    min_T = 0.1
    
    # Pre-compute indices for differences in a row of length k+1
    # There are (k+1)*k//2 differences
    diff_indices = []
    for i in range(k + 1):
        for j in range(i + 1, k + 1):
            diff_indices.append((i, j))
            
    # Optimization: Store current used diffs and row diffs to update incrementally?
    # For simplicity and robustness, we recalculate cost from scratch in the inner loop 
    # but optimize the calculation.
    # Actually, recalculating 112 diffs is very fast.
    
    iterations = 0
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Pick random row and index
        r_idx = random.randint(0, n_rows - 1)
        # Pick random index to change, not the first one (0)
        pos = random.randint(1, k)
        
        old_val = best_rows[r_idx][pos]
        
        # Determine valid range for new value
        lower = best_rows[r_idx][pos - 1] + 1
        upper = best_rows[r_idx][pos + 1] - 1 if pos < k else cur_max + 50 # Allow expansion
        
        if lower > upper:
            continue
            
        # Sample new value
        # Bias towards smaller values to reduce scope
        # But we need to explore.
        new_val = random.randint(lower, upper)
        
        if new_val == old_val:
            continue
            
        # Apply move
        best_rows[r_idx][pos] = new_val
        
        # Calculate new cost
        new_cost, new_max, new_conf = calculate_cost(best_rows)
        
        # Acceptance criterion
        delta = new_cost - cur_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            cur_cost = new_cost
            cur_max = new_max
            cur_conf = new_conf
            
            # Update best if valid and better
            if new_conf == 0 and new_cost < best_cost:
                best_cost = new_cost
                best_rows_copy = [row[:] for row in best_rows]
        else:
            # Revert
            best_rows[r_idx][pos] = old_val
            
        T *= cooling
        iterations += 1
        
        # Periodic check for time
        if iterations % 1000 == 0:
             if time.time() - start_time > time_limit:
                 break

    # Return the best valid solution found
    # If no valid solution was found (should not happen as greedy is valid), return greedy
    # We track best_rows_copy which is only updated when a valid solution with lower cost is found.
    # However, the initial greedy solution is valid.
    # So best_rows_copy is at least the greedy solution.
    
    return best_rows_copy
