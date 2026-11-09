import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Initialize rows using the greedy method (provides a valid baseline)
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
    
    # Calculate initial objective
    def calc_obj(rows):
        diffs = set()
        penalty = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        penalty += 1
                    else:
                        diffs.add(d)
        # Scope is the max of all entries
        scope = max(max(r) for r in rows)
        # We want to minimize scope, but penalties are critical for validity.
        # A large weight ensures we prioritize finding a valid solution first,
        # then minimize scope.
        # However, if we just minimize scope, we might get stuck in invalid states.
        # Let's use: Objective = scope + 1000 * penalty
        return scope + 1000 * penalty

    best_rows = [r[:] for r in rows]
    best_obj = calc_obj(best_rows)
    
    # Local Search / Simulated Annealing
    current_rows = [r[:] for r in rows]
    current_obj = calc_obj(current_rows)
    
    # Temperature parameters
    T = 50.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random row and a random position to modify
        r_idx = random.randint(0, n_rows - 1)
        p_idx = random.randint(1, k)  # Don't change the first element (0)
        
        # Generate a new value for current_rows[r_idx][p_idx]
        # The new value must be strictly greater than the previous element
        # and strictly less than the next element (if it exists)
        min_val = current_rows[r_idx][p_idx - 1] + 1
        max_val = current_rows[r_idx][p_idx + 1] - 1 if p_idx < k else current_rows[r_idx][p_idx] + 20
        
        if min_val > max_val:
            continue
            
        # Randomly perturb the value
        # Sometimes jump, sometimes small step
        if random.random() < 0.5:
            new_val = random.randint(min_val, max_val)
        else:
            step = random.choice([-2, -1, 1, 2, 3, 5, 10])
            new_val = current_rows[r_idx][p_idx] + step
            new_val = max(min_val, min(new_val, max_val))
            
        if new_val == current_rows[r_idx][p_idx]:
            continue
            
        old_val = current_rows[r_idx][p_idx]
        current_rows[r_idx][p_idx] = new_val
        
        new_obj = calc_obj(current_rows)
        
        # Acceptance criterion
        if new_obj < current_obj:
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_rows = [r[:] for r in current_rows]
        else:
            # Metropolis criterion
            delta = new_obj - current_obj
            if T > 0:
                prob = 2.718281828 ** (-delta / T)
                if random.random() < prob:
                    current_obj = new_obj
            else:
                current_rows[r_idx][p_idx] = old_val
                
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6
            
        # Periodic restart if stuck? 
        # For now, simple cooling. If we hit a local minimum with high penalty, 
        # the high T might not be enough. 
        # But since we start from a valid greedy solution, penalty is 0.
        # We are mainly trying to reduce scope.
        
    return best_rows
