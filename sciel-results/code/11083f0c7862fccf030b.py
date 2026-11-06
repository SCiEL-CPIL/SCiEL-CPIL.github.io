import math
import time

def construct():
    # Deadline guard: 120s limit, use 110s to be safe
    deadline = time.time() + 110
    
    selected_classes = []
    
    # Iterate through moduli in increasing order to prioritize small m (large 1/m)
    # We want to maximize sum(1/m), so smaller m is better.
    # Greedy: for each m, if we can find ANY valid residue r, we take it.
    for m in range(20, 201):
        if time.time() > deadline:
            break
        
        # Try to find a valid residue r for this m
        # r must be in [0, m)
        # Condition: for all (m2, r2) in selected_classes, r % gcd(m, m2) != r2 % gcd(m, m2)
        
        found_valid_r = None
        for r in range(m):
            is_valid = True
            for m2, r2 in selected_classes:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    is_valid = False
                    break
            if is_valid:
                found_valid_r = r
                break
        
        if found_valid_r is not None:
            selected_classes.append([m, found_valid_r])
            
    return selected_classes
