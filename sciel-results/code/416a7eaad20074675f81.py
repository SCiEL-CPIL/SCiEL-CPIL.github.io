"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import log

def compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def construct():
    start_time = time.time()
    time_limit = 115
    
    U = [0]
    best_score = compute_score(U)
    
    max_steps = 20
    search_width = 100
    
    for _ in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        best_x = None
        best_local_score = -1
        current_max = max(U)
        
        # Search for the next element
        # We prioritize smaller x to keep denominator small,
        # but we must check score.
        for x in range(current_max + 1, current_max + search_width + 1):
            U_new = U + [x]
            score = compute_score(U_new)
            if score > best_local_score:
                best_local_score = score
                best_x = x
        
        if best_x is None:
            break
            
        if best_local_score > best_score:
            U.append(best_x)
            best_score = best_local_score
        else:
            # If no improvement, try a wider range or stop?
            # If the best local score is not better than global best, 
            # adding this element hurts. 
            # We might try a larger jump.
            found = False
            for x in range(current_max + search_width + 1, current_max + search_width * 3 + 1):
                if time.time() - start_time > time_limit:
                    break
                U_new = U + [x]
                score = compute_score(U_new)
                if score > best_score:
                    best_local_score = score
                    best_x = x
                    found = True
                    break
            
            if found:
                U.append(best_x)
                best_score = best_local_score
            else:
                break
                
    return U
# EVOLVE-BLOCK-END
