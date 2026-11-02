"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s limit
    
    # Generate all weight-4 vectors with ±1 entries
    # C(11,4) = 330 positions, 2^4 = 16 sign combos -> 5280 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Norm squared of each candidate = 4
    # Constraint: |u-v|^2 >= 4 for all pairs, i.e., u·v <= 2
    
    # Greedy selection
    selected = []
    selected_norms = []
    
    for i in range(n_candidates):
        if time.time() > deadline - 5.0:
            break
        
        v = candidates[i]
        if not selected:
            selected.append(v)
            selected_norms.append(4)
            continue
        
        # Check constraint against all selected
        # |v - u|^2 = 4 + 4 - 2*v·u >= 4  =>  v·u <= 2
        dots = np.dot(selected, v)
        if np.all(dots <= 2):
            selected.append(v)
            selected_norms.append(4)
    
    if not selected:
        # Fallback: return a minimal valid set
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
