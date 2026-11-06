import time
import numpy as np

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Get all divisors of base greater than 1
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    
    # State
    covered = np.zeros(base, dtype=bool)
    classes = []
    remaining_moduli = set(divisors)
    
    # Precompute residue lists for each modulus? 
    # Or just compute on the fly.
    # For speed, we can precompute the indices for each (m, r)?
    # No, m is up to 17325, r up to m. Total indices is sum(m) ~ 120k.
    # But we only pick one r per m.
    
    # Greedy loop
    while remaining_moduli:
        if time.time() > deadline:
            break
            
        # Get uncovered indices
        uncovered = np.where(~covered)[0]
        if len(uncovered) == 0:
            break
            
        best_m = None
        best_r = None
        max_cov = -1
        
        # For each remaining modulus, find the residue that covers most uncovered
        # We iterate through moduli. 
        # To make it deterministic and potentially faster, we can sort moduli by size?
        # Smaller moduli have more classes per modulus? No, we pick best r for each m.
        # The cost of bincount is proportional to len(uncovered).
        # So all moduli cost the same per step.
        
        for m in remaining_moduli:
            # Counts of uncovered residues modulo m
            # np.bincount is fast
            counts = np.bincount(uncovered % m, minlength=m)
            current_max = np.max(counts)
            if current_max > max_cov:
                max_cov = current_max
                best_m = m
                best_r = int(np.argmax(counts))
                
        if max_cov == 0:
            # No more coverage possible
            break
            
        # Add the class
        classes.append([best_m, best_r])
        remaining_moduli.remove(best_m)
        
        # Update covered
        covered[best_r::best_m] = True
        
    return classes
