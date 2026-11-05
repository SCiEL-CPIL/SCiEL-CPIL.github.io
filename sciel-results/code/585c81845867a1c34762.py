import numpy as np
import time

def construct():
    n = 512
    best_size = 0
    best_set = np.zeros(n+1, dtype=bool)
    start_time = time.time()
    time_limit = 55  # Leave 5s margin
    
    # Precompute the triples for each x
    # triples[x] is a list of (a, b, c) such that a, b, c, x form a 4-AP
    # Actually, we just need the indices.
    
    while time.time() - start_time < time_limit:
        perm = np.random.permutation(1, n+1)
        current_set = np.zeros(n+1, dtype=bool)
        
        for x in perm:
            max_d = (x-1) // 3
            if max_d > 0:
                ds = np.arange(1, max_d + 1)
                a = x - 3 * ds
                b = x - 2 * ds
                c = x - ds
                # Check if any of these triples are all in current_set
                if np.any(current_set[a] & current_set[b] & current_set[c]):
                    continue
            current_set[x] = True
            
        size = np.sum(current_set)
        if size > best_size:
            best_size = size
            best_set = current_set.copy()
            
    return [i for i in range(1, n+1) if best_set[i]]
