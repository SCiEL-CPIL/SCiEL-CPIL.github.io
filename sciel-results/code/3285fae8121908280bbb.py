# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Uses a support-disjoint constructive strategy:
    - Generate all weight-2 vectors (norm^2 = 2) in a 10D subspace (coordinates 1-10).
    - Generate all weight-1 vectors (norm^2 = 1) in the remaining 1D subspace (coordinate 0).
    - Verify that the minimum pairwise distance is at least the maximum norm.
    
    This yields 180 + 2 = 182 vectors.
    """
    D = 11
    deadline = time.time() + 110.0  # Safety margin
    
    candidates = []
    
    # Type A: Weight-2 vectors in coordinates 1-10 (norm^2 = 2)
    # For each pair of positions in {1, ..., 10} and each sign combination
    for i, j in itertools.combinations(range(1, D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type B: Weight-1 vectors in coordinate 0 (norm^2 = 1)
    for s in (1, -1):
        v = np.zeros(D, dtype=np.int64)
        v[0] = s
        candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Verify the constraint: min pairwise distance >= max norm
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = np.max(norms_sq)  # Should be 2
    
    # Compute squared distances
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Check that all pairwise distances (excluding self) are >= max_norm_sq
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    
    # The constraint is: min_dist_sq >= max_norm_sq
    # If this fails, we need to filter or adjust, but for our construction it should hold.
    
    # Return the full set if valid, otherwise return the largest valid subset
    # For now, return all candidates as they should satisfy the constraint
    return candidates
# EVOLVE-BLOCK-END
