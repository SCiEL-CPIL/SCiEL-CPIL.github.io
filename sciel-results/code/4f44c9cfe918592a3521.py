"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    We use a randomized greedy algorithm to find a large independent set in the
    conflict graph of integer vectors with squared norm 4.
    The condition for two vectors u, v with ||u||^2 = ||v||^2 = 4 is
    ||u-v||^2 >= 4, which simplifies to <u,v> <= 2.
    """
    D = 11
    candidates = []
    
    # Type A: One ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            candidates.append(v)
            
    # Type B: Four ±1, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    best_set = []
    max_size = 0
    
    # Multiple random restarts to find a larger independent set
    for trial in range(200):
        random.seed(trial)
        indices = list(range(n_cands))
        random.shuffle(indices)
        
        selected = []
        selected_vecs = []
        
        for idx in indices:
            c = candidates[idx]
            if not selected:
                selected.append(idx)
                selected_vecs.append(c)
                continue
                
            # Vectorized dot product check
            # We need <c, s> <= 2 for all s in selected
            dots = np.dot(selected_vecs, c)
            if np.all(dots <= 2):
                selected.append(idx)
                selected_vecs.append(c)
                
        if len(selected) > max_size:
            max_size = len(selected)
            best_set = selected
            
    result = candidates[best_set]
    return result
# EVOLVE-BLOCK-END
