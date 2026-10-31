import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    # Key insight: for any two DISTINCT weight-4 vectors in {-1,0,1}^11,
    # their dot product is at most 2 (dot=4 implies identical vectors).
    # Since the compatibility condition is dot <= 2, ALL 5280 are mutually compatible.
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    # These are compatible with all weight-4 vectors (max dot = 2)
    # And among themselves: distinct pairs have dot <= 2 (dot=-4 for opposite signs same coord,
    # dot=0 for different coords). All 22 can be included.
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Quick sanity check: verify no conflicts in a sample of weight-4 vectors
    # and verify weight-1 compatibility
    if len(vectors_v4) > 0:
        sample_size = min(200, len(vectors_v4))
        sample = vectors_v4[:sample_size]
        dots_sample = sample @ sample.T
        mask = ~np.eye(sample_size, dtype=bool)
        if np.any(dots_sample[mask] > 2):
            # Fallback: should not happen, but handle gracefully
            # Use greedy MIS as safety net
            return _construct_with_mis(vectors_v4, vectors_v1, deadline)
    
    # Verify weight-1 vectors are compatible with each other
    if len(vectors_v1) > 0:
        dots_v1 = vectors_v1 @ vectors_v1.T
        mask_v1 = ~np.eye(len(vectors_v1), dtype=bool)
        if np.any(dots_v1[mask_v1] > 2):
            # Filter incompatible weight-1 vectors
            conflicts_v1 = dots_v1 > 2
            np.fill_diagonal(conflicts_v1, False)
            # Simple greedy: take all that don't conflict
            keep = np.ones(len(vectors_v1), dtype=bool)
            for i in range(len(vectors_v1)):
                if keep[i]:
                    conflicts_v1[i] = False
                    # Mark conflicting as removed
                    for j in range(i+1, len(vectors_v1)):
                        if keep[j] and dots_v1[i,j] > 2:
                            keep[j] = False
            vectors_v1 = vectors_v1[keep]
    
    final_vectors = np.vstack([vectors_v4, vectors_v1])
    return final_vectors

def _construct_with_mis(vectors_v4, vectors_v1, deadline):
    """Fallback MIS solver if the mathematical assumption is violated."""
    N4 = len(vectors_v4)
    if N4 == 0:
        return vectors_v1
        
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    alive = np.ones(N4, dtype=bool)
    selected_indices = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    rng = np.random.default_rng(42)
    
    while np.any(alive) and time.time() < deadline:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected_indices.append(chosen)
        
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        alive[to_remove] = False
        
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
            
        if time.time() > deadline - 10:
            break

    result_v4 = vectors_v4[selected_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
