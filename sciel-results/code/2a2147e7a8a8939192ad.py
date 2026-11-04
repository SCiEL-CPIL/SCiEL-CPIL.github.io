"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Generate all weight 4 vectors in Z^11
    # Coordinates are 0, 1, -1. Exactly 4 non-zeros.
    vectors = []
    indices = range(D)
    
    # Generate all combinations of 4 indices
    for combo in itertools.combinations(indices, 4):
        # Generate all sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            vectors.append(v)
            
    vectors = np.array(vectors, dtype=np.int64)
    
    # We need to find a subset of size 504 such that pairwise dot product <= 2.
    # Greedy approach with random shuffling.
    # To speed up, we can use numpy for batch checks if possible, 
    # but greedy with early termination is usually fast enough for 504 out of 5280.
    
    n_vectors = len(vectors)
    indices_shuffled = np.random.permutation(n_vectors)
    
    selected = []
    selected_vectors = []
    
    # Optimization: Pre-compute dot products is O(N^2) memory/time.
    # N=5280. N^2 = 27M. 
    # 27M operations is fast in numpy.
    # Let's compute the full dot product matrix?
    # Dot product matrix D_mat[i, j] = vectors[i] . vectors[j]
    # We need D_mat[i, j] <= 2 for all i != j in selected.
    
    # Computing full matrix:
    # D_mat = vectors @ vectors.T
    # This takes some time but is vectorized.
    
    D_mat = vectors @ vectors.T
    
    # Now greedily select
    # We want to pick indices from indices_shuffled
    # such that D_mat[sel, sel] <= 2 (off-diagonal)
    
    # Since D_mat is symmetric, we can just check against existing selected.
    
    for idx in indices_shuffled:
        if time.time() > deadline:
            break
            
        if len(selected) >= 504:
            break
            
        # Check compatibility with all selected
        # selected is a list of indices
        if not selected:
            selected.append(idx)
            continue
            
        # Get dot products of idx with all selected
        # D_mat[idx, selected]
        # We need all values <= 2
        # Note: D_mat[idx, idx] is 4, but we skip self-comparison
        
        dots = D_mat[idx, selected]
        if np.all(dots <= 2):
            selected.append(idx)
            
    # If we didn't reach 504, we might try a different shuffle or strategy
    # But 504 is likely reachable with random greedy on such a dense set.
    
    result = vectors[selected]
    
    # Verify validity (optional but good for debugging)
    # if len(result) > 0:
    #     norms = np.sum(result**2, axis=1)
    #     max_norm = np.max(norms)
    #     # Compute min distance
    #     # d^2 = ||v||^2 + ||w||^2 - 2 v.w
    #     # Since all norms are 4, d^2 = 8 - 2 v.w
    #     # We ensured v.w <= 2, so d^2 >= 4.
    #     # max_norm is 4.
    #     # 4 >= 4 is true.
    
    return result
# EVOLVE-BLOCK-END
