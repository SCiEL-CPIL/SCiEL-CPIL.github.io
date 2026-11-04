"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    
    # Generate all weight-4 vectors with entries +/- 1
    # There are C(11, 4) * 2^4 = 330 * 16 = 5280 such vectors
    candidates = []
    indices = range(D)
    
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    N = len(candidates)
    
    # Precompute dot product matrix
    # dots[i, j] = candidates[i] . candidates[j]
    # Condition for validity: dots[i, j] <= 2 for all i != j in selected
    # Note: dots[i, i] = 4
    
    # 5280 x 5280 matrix of int64 is ~223 MB. Acceptable.
    dots = candidates @ candidates.T
    
    selected = []
    
    # Greedy selection
    # Iterate through candidates. Add if compatible with all previously selected.
    for i in range(N):
        # Time check
        if time.time() - start_time > 100:
            break
            
        if not selected:
            selected.append(i)
            continue
            
        # Check compatibility with all selected vectors
        # selected is a list of indices
        # We need dots[i, selected] to be all <= 2
        
        # Using numpy fancy indexing
        valid_dots = dots[i, selected]
        
        if np.any(valid_dots > 2):
            continue
            
        selected.append(i)
        
    # Return the selected vectors
    if not selected:
        # Fallback to seed if something goes wrong (should not happen)
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return candidates[selected]
# EVOLVE-BLOCK-END
