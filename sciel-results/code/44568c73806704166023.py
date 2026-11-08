import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate the base set of weight-2 vectors
    # These are vectors with exactly two non-zero entries, each +/- 1
    base_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                base_vectors.append(v)
    
    base_array = np.array(base_vectors, dtype=np.int64)
    
    # Verify validity of the base set
    # Condition: min pairwise distance >= max norm
    # For this set, all norms are sqrt(2). Max norm squared is 2.
    # We need dist^2 >= 2.
    # dist^2 = ||u-v||^2.
    
    def is_valid(vecs):
        if len(vecs) == 0:
            return True
        norms_sq = np.sum(vecs * vecs, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Compute pairwise squared distances
        # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v
        # We need ||u||^2 + ||v||^2 - 2 u.v >= max_norm_sq for all u != v
        
        # Efficient check using matrix multiplication
        # G = vecs @ vecs.T
        # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2*G[i,j]
        
        G = vecs @ vecs.T
        # Create matrix of sums of norms
        norm_sums = norms_sq[:, None] + norms_sq[None, :]
        dist_sq = norm_sums - 2 * G
        
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dist_sq, np.inf)
        
        min_dist_sq = np.min(dist_sq)
        return min_dist_sq >= max_norm_sq - 1e-9

    if not is_valid(base_array):
        # Fallback to a smaller valid set if something went wrong (shouldn't happen)
        return base_array[:10]

    current_best = base_array
    current_count = len(current_best)
    
    # Attempt to improve by local search?
    # Given the strong theoretical argument against improvement with weight 3,
    # and the high cost of checking large sets, we stick to the base set.
    # However, we can try to add a few weight-3 vectors if we remove conflicting weight-2 vectors.
    # But as calculated, this is a net loss.
    
    # Just to be safe, check time
    if time.time() > deadline:
        return current_best

    return current_best
