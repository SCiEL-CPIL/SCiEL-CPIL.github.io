import time
import random

def construct():
    N = 51975
    # Find divisors
    divisors = []
    for i in range(2, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.append(N)
    # Ensure distinct and > 1
    divisors = sorted(set(divisors))
    
    # Precompute masks
    # masks[m] is a list of bitmasks for residues 0..m-1
    masks = {}
    for m in divisors:
        m_masks = []
        for r in range(m):
            mask = 0
            # Optimization: build mask
            # x = r + k*m
            for x in range(r, N, m):
                mask |= (1 << x)
            m_masks.append(mask)
        masks[m] = m_masks

    # Initial random assignment
    def random_assignment():
        return {m: random.randint(0, m - 1) for m in divisors}

    def get_score(assignment):
        union_mask = 0
        for m in divisors:
            r = assignment[m]
            union_mask |= masks[m][r]
        return N - union_mask.bit_count()

    best_score = N
    best_assignment = random_assignment()
    best_score = get_score(best_assignment)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        current_assignment = best_assignment.copy()
        current_score = best_score
        
        # Run a local search / annealing cycle
        # We can just do simple hill climbing with random restarts if stuck
        # Or a single long annealing run with restarts on plateau
        
        # Let's do a batch of moves
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            # Pick random modulus
            m = random.choice(divisors)
            old_r = current_assignment[m]
            new_r = random.randint(0, m - 1)
            
            if old_r == new_r:
                continue
                
            # Calculate delta score
            # Remove old mask, add new mask
            # To be fast, we can just recalculate total score
            # But delta is better if we can do it.
            # Union mask is not easily updatable for delta without storing the union.
            # Let's store the current union mask.
            
            # Actually, let's just recalculate score for simplicity first, 
            # but optimize by keeping the union mask in the loop.
            
            # Let's refactor to keep union mask
            pass
            
        # This inner loop is messy. Let's do a cleaner structure.
        
        # Break out of inner loop structure to restart cleanly
        break 

    # Revised Search Loop
    # We will run multiple independent annealing runs or just one long one.
    # Given the small search space (40 variables), a single long SA run with 
    # occasional restarts if stuck is good.
    
    current_assignment = random_assignment()
    current_score = get_score(current_assignment)
    if current_score < best_score:
        best_score = current_score
        best_assignment = current_assignment.copy()
        
    T = 5.0
    cooling = 0.9995
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        m = random.choice(divisors)
        old_r = current_assignment[m]
        new_r = random.randint(0, m - 1)
        
        if old_r == new_r:
            continue
            
        # Temporarily change
        current_assignment[m] = new_r
        new_score = get_score(current_assignment)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_assignment = current_assignment.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            current_assignment[m] = old_r
            no_improve_count += 1
            
        T *= cooling
        if T < 0.01:
            T = 0.01
            
        # If stuck for too long, restart
        if no_improve_count > max_no_improve:
            current_assignment = random_assignment()
            current_score = get_score(current_assignment)
            T = 5.0
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_assignment = current_assignment.copy()

    # Convert assignment to required format [[m, r], ...]
    result = []
    for m in divisors:
        result.append([m, best_assignment[m]])
        
    return result
