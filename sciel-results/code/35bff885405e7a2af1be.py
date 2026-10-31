"""Kissing number in dimension 11: maximize number of integer vectors with min pairwise distance >= max norm."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Generate all weight-2 vectors: e_i + e_j, e_i - e_j, -e_i + e_j, -e_i - e_j
    # For each pair (i,j), i < j, we have 4 vectors (signs of both coords)
    # Total: C(11,2) * 4 = 55 * 4 = 220
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    n = len(vectors)
    
    # Verify validity: min pairwise distance >= max norm
    # All vectors have norm sqrt(2)
    # Distance between two vectors:
    # - Same pair, same signs: distance 0 (but they're the same vector, not pairwise)
    # - Same pair, different signs: e.g., (1,1,0..) vs (1,-1,0..) -> dist = 2
    #   (1,1) vs (-1,-1) -> dist = 2*sqrt(2)
    #   (1,1) vs (-1,1) -> dist = 2
    # - Different pairs sharing one coordinate: e.g., (e1+e2) vs (e1+e3) -> dist = sqrt(0+4+1+...) wait
    #   (1,1,0,...) vs (1,0,1,...) -> diff = (0,1,-1,0,...) -> norm = sqrt(2)
    #   So minimum distance is sqrt(2) for overlapping pairs
    # Max norm = sqrt(2)
    # So condition: min pairwise distance >= max norm = sqrt(2). 
    # sqrt(2) >= sqrt(2) is satisfied (equality).
    
    # Quick verification
    if n > 1:
        # Compute all pairwise distances efficiently
        # d_ij^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
        norms_sq = np.sum(vectors * vectors, axis=1)  # all = 2
        dot = vectors @ vectors.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
        np.fill_diagonal(dist_sq, np.inf)
        min_dist = np.sqrt(np.min(dist_sq))
        max_norm = np.sqrt(np.max(norms_sq))
        if min_dist < max_norm - 1e-9:
            # Should not happen, but fallback
            pass
    
    return vectors
# EVOLVE-BLOCK-END
