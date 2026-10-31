"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_DEADLINE = 120.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - 5.0 - (time.time() - _START_TIME)


def construct():
    # We want to maximize k such that for all distinct i,j:
    #   ||v_i - v_j|| >= max_norm
    # Equivalently (for non-negative dot products in our constructions):
    #   v_i . v_j <= 0  (when all norms are equal)
    # More generally: ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j
    # We need: ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j >= max_norm^2
    # If all vectors have the same norm^2 = N, then: 2N - 2 v_i.v_j >= N  =>  v_i.v_j <= N/2

    # Strategy: Use vectors of norm^2 = 8 (weight-2 vectors with entries +/-2, or E8-related)
    # For weight-2 vectors with entries +/-1: norm^2 = 2, need v_i.v_j <= 1
    # For weight-2 vectors with entries +/-2: norm^2 = 8, need v_i.v_j <= 4

    # Let's use a mixed approach:
    # Group A: weight-2 vectors in first 8 dims with entries +/-1 (norm^2=2)
    # Group B: weight-2 vectors in first 8 dims with entries +/-2 (norm^2=8)
    # But mixing different norms complicates things. Let's use uniform norm.

    # Best approach from knowledge: 
    # - 220 weight-2 vectors in R^11 with entries +/-1 (norm^2=2)
    # - 128 E8 half-integer vectors scaled to integers (norm^2=8)
    # But these have different norms. Let's scale everything to norm^2=8.

    # Actually, let's use all vectors with norm^2 = 8:
    # Type 1: Two entries of +/-2, rest 0 (weight-2, magnitude 2)
    #   Number: C(11,2) * 4 = 55 * 4 = 220
    #   Dot product between two such vectors: 
    #     - Same position pair: 4 or -4 (if same signs) or 0 (if different pairs)
    #     - Wait, if they share exactly one position: 2*2=4 or 2*(-2)=-4
    #     - If they share both positions: 4 or -4
    #     - If they share no positions: 0
    #   For same position pair with same signs: dot = 4, which equals N/2 = 4. OK (boundary)
    #   For same position pair with opposite signs: dot = -4. OK.
    #   For sharing one position: dot = +/-4. OK.
    #   So all 220 are mutually compatible!

    vectors = []
    
    # Generate all weight-2 vectors with entries +/-2 in R^11
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
    
    # Now try to add more vectors. Let's try weight-4 vectors with entries +/-1
    # norm^2 = 4, need v_i.v_j <= 2
    # But mixing norm^2=8 and norm^2=4: 
    #   ||v8 - v4||^2 = 8 + 4 - 2*v8.v4 >= 8  =>  v8.v4 <= 2
    #   ||v4 - v4||^2 = 4 + 4 - 2*v4a.v4b >= 8  =>  v4a.v4b <= 0
    #   ||v8 - v8||^2 = 8 + 8 - 2*v8a.v8b >= 8  =>  v8a.v8b <= 4 (already satisfied)
    
    # This gets complex. Let's stick with the 220 weight-2 vectors first, then
    # greedily try to add more vectors of various types.
    
    vec_array = np.array(vectors, dtype=np.int64)
    n_base = len(vec_array)
    
    # Try to add weight-4 vectors with entries +/-1 (norm^2=4)
    # Need: dot with all weight-2 vectors <= 2, and dot with other weight-4 vectors <= 0
    added = 0
    for i, j, k, l in itertools.combinations(range(D), 4):
        if _time_left() < 10:
            break
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            v[i], v[j], v[k], v[l] = signs
            v_arr = np.array(v, dtype=np.int64)
            
            # Check against existing vectors
            dots = vec_array @ v_arr
            if np.max(dots) > 2:
                continue
            
            # Check if we can add it (also need to check against other weight-4 we'll add)
            # For simplicity, just check against current set
            vec_array = np.vstack([vec_array, v_arr])
            added += 1
    
    # Try to add weight-6 vectors with entries +/-1 (norm^2=6)
    # Need: dot with weight-2 (norm^2=8) <= (8+6-8)/2 = 3
    #       dot with weight-4 (norm^2=4) <= (4+6-8)/2 = 1
    #       dot with weight-6 (norm^2=6) <= (6+6-8)/2 = 2
    for i, j, k, l, m, n in itertools.combinations(range(D), 6):
        if _time_left() < 5:
            break
        for signs in itertools.product((1, -1), repeat=6):
            v = [0] * D
            v[i], v[j], v[k], v[l], v[m], v[n] = signs
            v_arr = np.array(v, dtype=np.int64)
            
            dots = vec_array @ v_arr
            if np.max(dots) > 3:
                continue
            
            vec_array = np.vstack([vec_array, v_arr])
    
    # Verify the result
    if len(vec_array) <= 1:
        return vec_array
    
    norms_sq = np.sum(vec_array ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Check all pairs
    # For efficiency, use the fact that we need ||v_i - v_j||^2 >= max_norm_sq
    # which is norms_sq[i] + norms_sq[j] - 2*v_i.v_j >= max_norm_sq
    
    # Compute all pairwise dot products
    dots = vec_array @ vec_array.T
    n = len(vec_array)
    
    # Check validity
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = norms_sq[i] + norms_sq[j] - 2 * dots[i, j]
            if dist_sq < max_norm_sq:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to just the 220 weight-2 vectors
        return np.array(vectors[:220], dtype=np.int64)
    
    return vec_array
# EVOLVE-BLOCK-END
