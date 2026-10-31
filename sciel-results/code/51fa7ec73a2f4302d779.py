import numpy as np
import time
import itertools
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # 1. Generate all weight-4 vectors with entries +/-1
    # Positions: choose 4 from 11
    # Signs: 2^4 = 16
    candidates = []
    for pos in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    # 2. Precompute dot product matrix
    # D[i, j] = dot(candidates[i], candidates[j])
    # We only need upper triangle or full? Full is easier.
    # Memory: 5280 * 5280 * 8 bytes ~ 224 MB. OK.
    dot_products = candidates @ candidates.T
    
    # 3. Add the 11 vectors 2e_i
    # These are compatible with all weight-4 vectors as proven.
    base_vectors = [2 * np.eye(D, dtype=np.int64)[i] for i in range(D)]
    
    best_count = 0
    best_set_indices = []
    
    # 4. Greedy search for max weight-4 set
    # We try multiple random orderings
    
    num_restarts = 0
    while time.time() < deadline:
        # Create a random permutation of indices
        order = np.random.permutation(N)
        
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            # Check if candidate idx is compatible with all selected
            if len(selected) == 0:
                selected.append(idx)
                selected_mask[idx] = True
            else:
                # Get dot products with selected
                # dot_products[idx, selected]
                # Check if max <= 2
                if np.max(dot_products[idx, selected]) <= 2:
                    selected.append(idx)
                    selected_mask[idx] = True
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_set_indices = selected.copy()
        
        num_restarts += 1
        
        # Optional: Local search improvement
        # Try to swap some vectors to allow more?
        # For now, just do many restarts.
        
    # Construct final array
    # 11 base vectors + best_count weight-4 vectors
    final_vectors = base_vectors.copy()
    for idx in best_set_indices:
        final_vectors.append(candidates[idx])
        
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-START
# The code above is inside construct()
# EVOLVE-BLOCK-END
