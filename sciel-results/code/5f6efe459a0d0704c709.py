import random
import math

def construct(n_rows, k):
    # 1. Greedy Initialization
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
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    # 2. Local Search / Simulated Annealing
    # We want to minimize the max value.
    
    rows, used = greedy_init()
    
    # Helper to get max value
    def get_max_val(rws):
        return max(max(r) for r in rws)
    
    current_max = get_max_val(rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # SA Parameters
    T = 100.0
    cooling = 0.99
    iterations = 50000
    random.seed(42) # For reproducibility in testing, but maybe remove for final? 
    # The task says "The evaluator passes this task's n_rows and k". 
    # It doesn't specify determinism, but consistent behavior is good.
    
    for _ in range(iterations):
        if T < 0.01:
            break
            
        # Pick a random row and element
        r = random.randint(0, n_rows - 1)
        j = random.randint(1, k)
        
        v = rows[r][j]
        
        # Determine range for new value
        min_v = rows[r][j-1] + 1
        max_v = rows[r][j+1] - 1 if j < k else current_max + 10 # Allow expansion if needed? 
        # Actually, we want to shrink. But to escape local minima, allow some expansion.
        # However, if we expand, max_val might increase.
        
        # Strategy: Try to move to a smaller value if possible, or nearby.
        # Let's pick a candidate v'
        # Bias towards smaller values to reduce max
        if v > min_v:
            # Try to reduce
            v_prime = random.randint(min_v, v)
        else:
            v_prime = v
            
        # If v_prime == v, skip
        if v_prime == v:
            continue
            
        # Check validity
        # 1. Remove old diffs
        old_diffs = set()
        for a in rows[r]:
            if a != v:
                d = abs(v - a)
                old_diffs.add(d)
        
        # 2. Calculate new diffs
        new_diffs = set()
        for a in rows[r]:
            if a != v:
                d = abs(v_prime - a)
                new_diffs.add(d)
        
        # 3. Check collisions
        # The global 'used' set contains all diffs.
        # We need to check if any diff in 'new_diffs' is in 'used' 
        # BUT we have to ignore the ones in 'old_diffs' because they are being removed.
        
        # Efficient check:
        # potential_collisions = new_diffs & used
        # valid_collisions = potential_collisions - old_diffs
        # if valid_collisions: collision exists.
        
        valid_collisions = new_diffs & (used - old_diffs)
        
        if valid_collisions:
            # Collision. Reject (or accept with low prob)
            # For simplicity, reject collisions to stay in valid space.
            # This is Hill Climbing on valid states.
            continue
        else:
            # Valid move
            # Calculate cost change
            old_max = get_max_val(rows)
            
            # Apply move
            rows[r][j] = v_prime
            used -= old_diffs
            used |= new_diffs
            
            new_max = get_max_val(rows)
            
            delta_max = new_max - old_max
            
            # We accept if it improves max, or with probability exp(-delta/T)
            # Since we are in valid space, delta_max is the only cost.
            if delta_max < 0:
                # Improvement
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
            else:
                # Worse or equal
                # Accept with probability exp(-delta_max / T)
                if random.random() < math.exp(-delta_max / T):
                    pass # Accept
                else:
                    # Reject, revert
                    rows[r][j] = v
                    used -= new_diffs
                    used |= old_diffs
            
            T *= cooling
            
    return best_rows
