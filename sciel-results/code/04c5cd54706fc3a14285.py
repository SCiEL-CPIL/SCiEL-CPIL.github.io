import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0
    
    # Helper to calculate cost
    def get_cost(rows):
        used = set()
        conflicts = 0
        max_val = 0
        for row in rows:
            max_val = max(max_val, max(row))
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
        # If there are conflicts, the cost is very high.
        # We want to minimize max_val only when conflicts are 0.
        # Cost = conflicts * LARGE + max_val
        return conflicts * 10000 + max_val

    # 1. Greedy Initialization
    def greedy_construct():
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
        return rows

    rows = greedy_construct()
    current_cost = get_cost(rows)
    best_rows = [list(r) for r in rows]
    best_cost = current_cost

    # 2. Simulated Annealing
    T = 100.0
    cooling = 0.99
    min_T = 0.1
    step_size = max(1, (best_rows[0][-1] // 10)) if best_rows else 1

    while T > min_T and time.time() - start_time < time_limit:
        # Generate a new candidate
        new_rows = [list(r) for r in rows]
        
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = new_rows[r_idx]
        
        # Pick a random index to change (1 to k)
        if len(row) > 1:
            j_idx = random.randint(1, len(row) - 1)
            old_val = row[j_idx]
            
            # Determine range for new value
            # Must be > row[j_idx-1] and < row[j_idx+1] (if exists)
            lower = row[j_idx-1] + 1
            upper = row[j_idx+1] - 1 if j_idx + 1 < len(row) else old_val + step_size
            
            # If range is invalid, try a different move or skip
            if lower > upper:
                # Try swapping with a neighbor or just skip this step
                continue
                
            # To encourage reduction of max value, bias towards lower values
            # but allow exploration.
            # Simple uniform random in [lower, upper]
            new_val = random.randint(lower, upper)
            
            if new_val == old_val:
                continue
                
            row[j_idx] = new_val
            
            # Check if row is still strictly increasing
            # It is by construction of lower/upper, but let's be safe.
            # Actually, if we pick j_idx, we only change one element.
            # We must ensure row[j_idx-1] < new_val < row[j_idx+1].
            # The bounds above ensure this.
            
            new_cost = get_cost(new_rows)
            
            if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / T):
                rows = new_rows
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [list(r) for r in rows]
        
        T *= cooling

    # Return the best valid solution found
    # We must ensure the returned solution is valid (0 conflicts).
    # If best_cost has conflicts (i.e., best_cost >= 10000), we should return the greedy one?
    # The greedy one is always valid.
    # So we check if best_cost < 10000.
    if best_cost < 10000:
        return best_rows
    else:
        return greedy_construct()
