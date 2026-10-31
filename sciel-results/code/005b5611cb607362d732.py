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
    # Minimum distance between distinct vectors in this set is sqrt(2)
    # (e.g., e_1+e_2 and e_1+e_3 have distance sqrt(2))
    # Max norm is sqrt(2)
    # Condition: min_dist >= max_norm holds with equality.
    
    # Quick verification (fixed to avoid OverflowError)
    if n > 1:
        # Compute all pairwise distances efficiently
        # d_ij^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
        norms_sq = np.sum(vectors * vectors, axis=1)  # all = 2
        dot = vectors @ vectors.T
        # Convert to float to allow inf and avoid integer overflow issues with inf
        dist_sq = (norms_sq[:, None] + norms_sq[None, :] - 2 * dot).astype(np.float64)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist = np.sqrt(np.min(dist_sq))
        max_norm = np.sqrt(np.max(norms_sq))
        if min_dist < max_norm - 1e-9:
            # Should not happen, but fallback
            pass
    
    # Check deadline (not strictly necessary for static generation, but kept for compliance)
    if time.time() > deadline:
        pass # Return whatever we have
        
    return vectors
# EVOLVE-BLOCK-END
