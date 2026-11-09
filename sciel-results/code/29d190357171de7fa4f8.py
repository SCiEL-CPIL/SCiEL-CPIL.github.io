import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    # Set a hard deadline to ensure we return within 60s
    # Precomputation + Search
    deadline = start_time + 55.0
    
    N = 16
    K = 4
    subsets = list(combinations(range(N), K))
    num_subsets = len(subsets)
    
    # Convert to bitmasks
    masks = []
    for s in subsets:
        m = 0
        for x in s:
            m |= (1 << x)
        masks.append(m)
    masks = np.array(masks, dtype=np.int32)
    
    # Precompute intersection matrix
    # M[i][j] = masks[i] & masks[j]
    M = masks[:, None] & masks[None, :]
    
    # Check time
    if time.time() > deadline - 5.0:
         # Fallback: return empty or a small valid set
         return [list(subsets[0])]

    def greedy(order):
        chosen = []
        for c in order:
            if time.time() > deadline:
                break
            ok = True
            n_chosen = len(chosen)
            # If n_chosen is large, we can precompute vals_c
            if n_chosen > 0:
                # Get intersection of c with all chosen
                # This creates a view or copy? M[chosen, c] is a 1D array
                vals_c = M[chosen, c]
                # M_sub is the submatrix of chosen x chosen
                # We don't want to copy M_sub every time.
                # But accessing M[chosen[i], chosen[j]] is slow.
                # Let's stick to index access for simplicity, it's fast enough for N=100.
                
                for i in range(n_chosen):
                    ai = chosen[i]
                    mai_c = vals_c[i]
                    for j in range(i + 1, n_chosen):
                        aj = chosen[j]
                        # Check M[ai][aj] == mai_c == M[aj][c]
                        # M[aj][c] is vals_c[j]
                        if M[ai][aj] == mai_c and vals_c[j] == mai_c:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                chosen.append(c)
        return chosen

    best_family = []
    
    # 1. Lexicographic
    order = list(range(num_subsets))
    fam = greedy(order)
    best_family = fam
    
    # 2. Random Restart
    # Estimate time
    t0 = time.time()
    # Do 5 runs to estimate
    for _ in range(5):
        if time.time() > deadline - 2.0:
            break
        random.shuffle(order)
        fam = greedy(order)
        if len(fam) > len(best_family):
            best_family = fam
    t1 = time.time()
    elapsed = t1 - t0
    if elapsed > 0:
        time_per_run = elapsed / 5
    else:
        time_per_run = 0.1
        
    # Run more
    remaining = deadline - time.time()
    num_runs = int(remaining / time_per_run)
    
    for _ in range(num_runs):
        if time.time() > deadline:
            break
        random.shuffle(order)
        fam = greedy(order)
        if len(fam) > len(best_family):
            best_family = fam
            
    # Return result
    return [sorted(subsets[i]) for i in best_family]
