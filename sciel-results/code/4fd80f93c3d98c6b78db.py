"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations


def _generate_weight4_candidates():
    """Generate all nonzero integer vectors of weight 4 in R^11."""
    candidates = []
    for pos in combinations(range(D), 4):
        pos_set = set(pos)
        for signs in range(16):  # 4 bits
            v = np.zeros(D, dtype=np.int64)
            for bit, p in enumerate(pos):
                sgn = 1 if (signs >> bit) & 1 else -1
                v[p] = sgn
            candidates.append(v)
    return candidates


def _distance_sq(u, v):
    """Squared Euclidean distance between two integer vectors."""
    return int(np.dot(u - v, u - v))


def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    candidates = _generate_weight4_candidates()
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.array([np.dot(v, v) for v in candidates], dtype=np.int64)
    max_norm_sq = int(norms_sq.max())

    # Build conflict matrix: conflict[i,j] = True if dist_sq(u_i, u_j) < max_norm_sq
    # For weight-4 vectors, norm_sq = 16, so we need dist_sq >= 16
    # dist_sq(u,v) = ||u||^2 + ||v||^2 - 2*u.v = 32 - 2*u.v
    # Need 32 - 2*u.v >= 16 => u.v <= 8
    # For weight-4 vectors, u.v ranges from -16 to 16 (integer)
    # So conflict if u.v > 8, i.e., u.v in {9, 10, 11, 12, 13, 14, 15, 16}
    
    # Compute dot product matrix
    cand_arr = np.array(candidates, dtype=np.int64)
    dot_matrix = cand_arr @ cand_arr.T
    
    # Conflict: dot > 8 (i.e., dist_sq < 16)
    conflict = dot_matrix > 8
    
    # Greedy set-packing: pick vectors that conflict with fewest remaining compatible partners
    # "Future conflict radius" = number of other candidates that conflict with this one
    # We want to pick low-conflict vectors first to preserve search space
    
    remaining = set(range(n_cand))
    selected = []
    
    # Initial conflict counts
    conflict_count = np.zeros(n_cand, dtype=np.int64)
    for i in range(n_cand):
        conflict_count[i] = np.sum(conflict[i])
    
    while remaining:
        if time.time() > deadline - 5.0:
            break
        
        # Find candidate in remaining with minimum conflict count
        best_idx = -1
        best_count = float('inf')
        for idx in remaining:
            if conflict_count[idx] < best_count:
                best_count = conflict_count[idx]
                best_idx = idx
        
        if best_idx == -1:
            break
        
        selected.append(best_idx)
        remaining.discard(best_idx)
        
        # Update conflict counts for remaining candidates
        # For each remaining j, if conflict[best_idx, j], decrement their count
        for j in list(remaining):
            if conflict[best_idx, j]:
                conflict_count[j] -= 1
    
    if not selected:
        # Fallback: return at least one vector
        return cand_arr[:1].copy()
    
    result = cand_arr[selected].copy()
    return result
# EVOLVE-BLOCK-END
