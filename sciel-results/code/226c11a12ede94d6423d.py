import numpy as np
import time
import random

def construct():
    base = 51975
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Get all divisors of base > 1
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    moduli = sorted(divisors)
    
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    
    # Greedy Initialization
    # Sort ascending: small moduli are more constrained
    for m in moduli:
        if time.time() > time_limit:
            break
        best_r = 0
        min_covered_count = base
        for r in range(m):
            cnt = covered[r::m].sum()
            if cnt < min_covered_count:
                min_covered_count = cnt
                best_r = r
                
        classes.append([m, best_r])
        covered[best_r::m] = 1
        
    current_uncovered = int(np.sum(covered == 0))
    best_uncovered = current_uncovered
    best_classes = [list(c) for c in classes]
    
    if best_uncovered == 0:
        return best_classes

    # Local Search
    improved = True
    
    # We will alternate between descending and ascending order of moduli 
    # to help escape local minima, or just use a fixed order. 
    # Descending size often allows larger moduli to "fill in" gaps.
    search_order_desc = sorted(range(len(moduli)), key=lambda i: -moduli[i])
    search_order_asc = sorted(range(len(moduli)), key=lambda i: moduli[i])
    
    pass_idx = 0
    while improved and (time.time() < time_limit):
        improved = False
        # Alternate order
        if pass_idx % 2 == 0:
            order = search_order_desc
        else:
            order = search_order_asc
        pass_idx += 1
        
        for idx in order:
            if time.time() > time_limit:
                break
                
            m = moduli[idx]
            old_r = classes[idx][1]
            
            # Clear the current contribution of this modulus
            covered[old_r::m] = 0
            
            # Calculate uncovered status without this modulus
            uncovered = (covered == 0)
            base_uncovered = int(np.sum(uncovered))
            
            best_r = old_r
            best_new_uncovered = base_uncovered
            
            # Try all residues
            # Vectorized sum for all r is not trivial due to strides, 
            # but loop is fast enough.
            for r in range(m):
                gain = int(np.sum(uncovered[r::m]))
                new_uncovered = base_uncovered - gain
                
                if new_uncovered < best_new_uncovered:
                    best_new_uncovered = new_uncovered
                    best_r = r
            
            # Apply the best residue
            if best_r != old_r:
                covered[best_r::m] = 1
                classes[idx][1] = best_r
                current_uncovered = best_new_uncovered
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [list(c) for c in classes]
                    improved = True
                    if best_uncovered == 0:
                        return best_classes
            else:
                # Restore old state
                covered[old_r::m] = 1
                
    return best_classes
