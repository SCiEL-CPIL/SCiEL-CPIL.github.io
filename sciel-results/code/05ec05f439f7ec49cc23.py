import numpy as np
import random
import time

def construct():
    base = 51975
    # Get all divisors > 1 of base
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    
    best_score = base
    best_classes = None
    
    start_time = time.time()
    time_limit = 100.0
    
    # Shuffle divisors for initialization order? 
    # The set of divisors is fixed. The order of adding them in greedy init matters.
    
    num_restarts = 0
    while time.time() - start_time < time_limit:
        num_restarts += 1
        
        # Greedy Initialization
        # Shuffle the order in which we add divisors to break symmetry
        init_order = divisors[:]
        random.shuffle(init_order)
        
        classes = {}
        for m in init_order:
            classes[m] = None # Placeholder
            
        cover_count = np.zeros(base, dtype=np.int32)
        
        # For each modulus, pick residue that covers most currently uncovered spots
        for m in init_order:
            # Find r that maximizes sum(cover_count[r::m] == 0)
            # Z is where cover_count is 0
            Z = (cover_count == 0)
            # Reshape and sum
            # base is divisible by m
            Z_reshaped = Z.reshape(-1, m)
            counts = Z_reshaped.sum(axis=0)
            r_best = np.argmax(counts)
            classes[m] = int(r_best)
            cover_count[r_best::m] += 1
            
        # Convert to list of lists for interface
        current_classes = [[m, classes[m]] for m in divisors]
        
        # Calculate initial score
        uncovered_count = np.sum(cover_count == 0)
        
        # Coordinate Descent
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                break
            improved = False
            for m in divisors:
                r_old = classes[m]
                
                # Temporarily remove this class
                cover_count[r_old::m] -= 1
                
                # Find best new residue
                Z = (cover_count == 0)
                Z_reshaped = Z.reshape(-1, m)
                counts = Z_reshaped.sum(axis=0)
                r_new = int(np.argmin(counts))
                min_zeros = int(counts[r_new])
                
                # Current zeros for r_old in this state
                # We need to know how many zeros were in r_old::m in the "removed" state
                # This is counts[r_old]
                current_zeros = int(counts[r_old])
                
                if min_zeros < current_zeros:
                    # Apply change
                    # We already decremented r_old::m
                    # Now increment r_new::m
                    cover_count[r_new::m] += 1
                    classes[m] = r_new
                    uncovered_count += min_zeros - current_zeros
                    improved = True
                else:
                    # Revert: we decremented r_old::m, but didn't change class
                    # So we must increment it back
                    cover_count[r_old::m] += 1
                    # classes[m] remains r_old
        
        # Check if better
        if uncovered_count < best_score:
            best_score = uncovered_count
            best_classes = [c[:] for c in current_classes]
            
        if best_score == 0:
            break

    return best_classes
