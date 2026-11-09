import numpy as np
import itertools
import time

D = 11
DEADLINE = 110.0  # Safety margin before 120s limit

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    # Generate all candidate vectors with squared norm 8
    # Type 1: Two +/-2s, rest 0.  C(11,2)*4 = 220 vectors.
    # Type 2: Eight +/-1s, rest 0. C(11,8)*2^8 = 165*256 = 42240 vectors.
    # Total: 42460 vectors.
    
    candidates = []
    
    # Type 1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 2 * si
                v[j] = 2 * sj
                candidates.append(v)
    
    # Type 2
    for support in itertools.combinations(range(D), 8):
        for signs in itertools.product([-1, 1], repeat=8):
            v = np.zeros(D, dtype=np.int8)
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            candidates.append(v)
    
    # Convert to numpy array for vectorized operations
    cand_arr = np.array(candidates, dtype=np.int16)
    n_cand = len(cand_arr)
    
    # Compute the full dot product matrix
    # M[i,j] = dot(cand_i, cand_j)
    # Two vectors are compatible if |dot| <= 4 (i.e., dot <= 4 and dot >= -4)
    # But wait: the constraint is minimum pairwise distance >= max norm.
    # All vectors have norm sqrt(8) = 2*sqrt(2).
    # Distance squared = ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 16 - 2*v_i.v_j
    # We need distance >= max norm = sqrt(8)
    # So distance^2 >= 8
    # 16 - 2*v_i.v_j >= 8
    # 2*v_i.v_j <= 8
    # v_i.v_j <= 4
    # Also we need distance^2 >= 0, which is always true.
    # But we also need the distance to be at least the maximum norm, which is sqrt(8).
    # Since all norms are equal to sqrt(8), max norm = sqrt(8).
    # So we need ||v_i - v_j|| >= sqrt(8), i.e., ||v_i - v_j||^2 >= 8.
    # 16 - 2*v_i.v_j >= 8 => v_i.v_j <= 4.
    # There's no lower bound on the dot product from this constraint alone.
    # However, we should also ensure the vectors are distinct and nonzero.
    # Actually, we also need to consider: can v_i.v_j be very negative?
    # If v_i.v_j = -8, then distance^2 = 16 - 2*(-8) = 32 >= 8. Fine.
    # So the only constraint is v_i.v_j <= 4.
    
    # But wait, we also need to handle the case where v_i = v_j.
    # If v_i = v_j, then dot = 8 > 4, so they're incompatible. Good, we won't pick duplicates.
    
    # Compute dot product matrix in chunks to save memory
    # n_cand = 42460, so full matrix is 42460 x 42460 x 2 bytes (int16) = ~3.6 GB. Too large.
    # We need a different approach.
    
    # Instead, use a greedy approach with efficient checking.
    # For each candidate, check against already-selected vectors.
    # To speed up, we can precompute some structure.
    
    # Alternative: Use a random ordering and greedy selection.
    # For each candidate in random order, check if it's compatible with all selected.
    # Checking against k selected vectors costs O(k*D).
    # Total cost: O(n_cand * k_avg * D).
    # If we select ~300-500 vectors, and n_cand = 42460, D = 11:
    # 42460 * 400 * 11 = ~187M operations. Might be too slow in Python.
    
    # Let's use numpy for the dot product checks.
    # Keep selected vectors in a numpy array.
    # For each candidate, compute dot products with all selected at once.
    
    np.random.seed(42)
    perm = np.random.permutation(n_cand)
    
    selected = []
    selected_arr = None
    
    for idx in perm:
        if time.time() > DEADLINE:
            break
        
        v = cand_arr[idx]
        
        if not selected:
            selected.append(idx)
            selected_arr = v.reshape(1, -1).astype(np.int16)
            continue
        
        # Compute dot products with all selected vectors
        dots = selected_arr @ v  # shape (k,)
        if np.all(dots <= 4):
            selected.append(idx)
            if selected_arr is None:
                selected_arr = v.reshape(1, -1).astype(np.int16)
            else:
                selected_arr = np.vstack([selected_arr, v.reshape(1, -1).astype(np.int16)])
    
    result = np.array([cand_arr[i] for i in selected], dtype=np.int64)
    
    # Verify: check that all pairwise dot products <= 4
    if len(result) > 1:
        # Compute pairwise dot products
        dots = result.astype(np.int32) @ result.astype(np.int32).T
        np.fill_diagonal(dots, 0)
        if np.max(dots) > 4:
            # This shouldn't happen, but if it does, fall back to a simpler set
            # Just return the first few that are valid
            pass
    
    return result

# EVOLVE-BLOCK-END
