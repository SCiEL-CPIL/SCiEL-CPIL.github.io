import random
import math
import time

def construct(n_rows, k):
    # 1. Greedy Initialization
    def greedy_init(seed_val=None):
        if seed_val is not None:
            random.seed(seed_val)
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
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    # Helper to get max value
    def get_max_val(rws):
        return max(max(r) for r in rws)

    # SA Parameters
    start_time = time.time()
    timeout = 110.0  # Leave 10s buffer for evaluation overhead
    
    best_rows = None
    best_max = float('inf')
    
    # Try a few different random seeds to escape local minima
    # Or just one long run. Let's do multiple restarts if time permits.
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout - 5:
            break
            
        # Randomize initialization slightly by shuffling the order of rows or using different seeds
        # The greedy init is deterministic for a given seed.
        rows, used = greedy_init(seed_val=42 + restart)
        
        current_max = get_max_val(rows)
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in rows]
            
        T = 50.0  # Start with lower temperature for refinement
        cooling = 0.995
        iterations = 20000  # More iterations per restart
        
        for _ in range(iterations):
            if time.time() - start_time > timeout:
                break
            if T < 0.01:
                break
                
            # Pick a random row and element
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            
            v = rows[r][j]
            
            # Determine range for new value
            min_v = rows[r][j-1] + 1
            # Allow expansion up to current_max + 10 to escape local minima
            max_v = max(current_max + 10, v + 10)
            if j < k:
                max_v = min(max_v, rows[r][j+1] - 1)
            
            if min_v > max_v:
                continue
                
            # Bias towards smaller values to reduce max, but allow larger
            # Use a biased random choice
            # Probability of choosing a value v' is proportional to exp(-v'/T_bias) ?
            # Simpler: 80% chance to pick from [min_v, v], 20% from [v, max_v]
            if random.random() < 0.8 and v > min_v:
                v_prime = random.randint(min_v, v)
            else:
                v_prime = random.randint(min_v, max_v)
                
            if v_prime == v:
                continue
            
            # Check validity
            old_diffs = set()
            for a in rows[r]:
                if a != v:
                    d = abs(v - a)
                    old_diffs.add(d)
            
            new_diffs = set()
            for a in rows[r]:
                if a != v:
                    d = abs(v_prime - a)
                    new_diffs.add(d)
            
            # Check collisions
            # The global 'used' set contains all diffs.
            # We need to check if any diff in 'new_diffs' is in 'used' 
            # BUT we have to ignore the ones in 'old_diffs' because they are being removed.
            valid_collisions = new_diffs & (used - old_diffs)
            
            if valid_collisions:
                continue
            
            # Valid move
            old_max = current_max
            
            # Apply move
            rows[r][j] = v_prime
            used -= old_diffs
            used |= new_diffs
            
            new_max = get_max_val(rows)
            
            delta_max = new_max - old_max
            
            # Acceptance criterion
            if delta_max < 0:
                # Improvement
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
            else:
                # Worse or equal
                # Accept with probability exp(-delta_max / T)
                if random.random() < math.exp(-delta_max / T):
                    current_max = new_max
                else:
                    # Reject, revert
                    rows[r][j] = v
                    used -= new_diffs
                    used |= old_diffs
                    # current_max remains old_max
            
            T *= cooling

    if best_rows is None:
        # Fallback to greedy if no SA run completed
        rows, _ = greedy_init(42)
        return rows
        
    return best_rows
