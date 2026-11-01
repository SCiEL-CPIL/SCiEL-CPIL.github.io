import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    # Generate all weight-2 vectors in {-1, 0, 1}^11 (Norm^2 = 2)
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_candidates.append(v)

    all_vectors = np.vstack([
        np.array(v4_candidates, dtype=np.int64),
        np.array(v1_candidates, dtype=np.int64),
        np.array(v2_candidates, dtype=np.int64)
    ])
    
    N = len(all_vectors)
    norms_sq = np.zeros(N, dtype=np.int64)
    n4 = len(v4_candidates)
    n1 = len(v1_candidates)
    n2 = len(v2_candidates)
    norms_sq[:n4] = 4
    norms_sq[n4:n4+n1] = 4
    norms_sq[n4+n1:] = 2
    
    # Compute dot products
    dots = all_vectors @ all_vectors.T
    
    # Conflict if dot > min(norm_i, norm_j) / 2
    min_norms = np.minimum.outer(norms_sq, norms_sq)
    thresholds = min_norms / 2.0
    conflicts = dots > thresholds
    np.fill_diagonal(conflicts, False)
    
    # Greedy MIS
    rng = np.random.default_rng(42)
    alive = np.ones(N, dtype=bool)
    selected = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    while np.any(alive):
        if time.time() > deadline - 10:
            break
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        selected.append(chosen)
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        alive[to_remove] = False
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
            
    result = all_vectors[selected]
    return result
