import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    candidates = []
    
    # Part 1: Weight-2 vectors in {-1, 0, 1}^D (norm^2 = 2)
    # C(11,2) * 4 = 220 vectors
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Part 2: E8 lattice vectors embedded in first 8 dimensions
    # E8 minimal vectors have norm^2 = 2 in standard embedding
    # Scale by 1 to keep integer: norm^2 = 2
    # E8 has 240 minimal vectors
    # Generate E8 minimal vectors:
    # Type 1: permutations of (±1, ±1, 0,0,0,0,0,0) - C(8,2)*4 = 112
    # Type 2: permutations of (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs
    #   That's 2^7 = 128 vectors, each with norm^2 = 8*(1/4) = 2
    # Scale type 2 by 2 to get integers: norm^2 = 8, entries in {±1}
    
    # For consistency, let's use a different approach.
    # We want all vectors to have the same norm^2 so that the distance constraint is uniform.
    # Let's use norm^2 = 8 for all vectors.
    
    # Clear candidates and rebuild with norm^2 = 8
    candidates = []
    
    # Part 1: Weight-4 vectors in {-1, 0, 1}^D with norm^2 = 4... no that's norm^2=4.
    # Let me reconsider. 
    
    # Strategy: Use two groups with the same norm^2.
    # Group A: weight-2 vectors, norm^2 = 2. Scale by 2: norm^2 = 8, entries in {0, ±2}.
    # Group B: E8-type vectors in first 8 dims.
    
    # Actually, let's just use weight-2 vectors (norm^2=2) and add E8 half-integer vectors scaled up.
    # E8 half-integer vectors: 128 vectors, norm^2 = 2. Scale by 2: norm^2 = 8.
    # Weight-2 vectors: 220 vectors, norm^2 = 2. Scale by 2: norm^2 = 8.
    # 
    # For two vectors v, w with |v|^2 = |w|^2 = 8:
    # dist^2 = 16 - 2*v.w >= 8  =>  v.w <= 4
    #
    # For scaled weight-2 vectors: entries in {0, ±2}
    # For scaled E8 half-int: entries in {±1} in first 8 dims, 0 in last 3
    
    # Let's generate:
    
    # Group A: scaled weight-2 vectors (norm^2 = 8)
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                candidates.append(v)
    
    # Group B: E8 half-integer vectors scaled by 2 (norm^2 = 8)
    # These have entries ±1 in first 8 positions, 0 in last 3
    # Even number of -1's among the 8 entries
    for mask in range(256):
        # Count bits (number of -1's)
        num_neg = bin(mask).count('1')
        if num_neg % 2 != 0:
            continue
        v = [0] * D
        for k in range(8):
            if mask & (1 << k):
                v[k] = -1
            else:
                v[k] = 1
        # v[8], v[9], v[10] remain 0
        candidates.append(v)
    
    vectors = np.array(candidates, dtype=np.int64)
    N = len(vectors)
    
    if N == 0:
        return np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Verify constraint: min pairwise dist^2 >= max norm^2
    dots = vectors @ vectors.T
    norms2 = (vectors ** 2).sum(axis=1)
    max_norm2 = norms2.max()
    # dist^2(i,j) = norms2[i] + norms2[j] - 2*dots[i,j]
    dist2 = norms2[:, None] + norms2[None, :] - 2 * dots
    np.fill_diagonal(dist2, 0)
    
    min_dist2 = dist2.min()
    
    if min_dist2 < max_norm2:
        # Need to find a maximal subset satisfying the constraint
        # Conflict: dist2[i,j] < max_norm2
        conflicts = dist2 < max_norm2
        np.fill_diagonal(conflicts, False)
        
        # Greedy MIS with degree heuristic
        alive = np.ones(N, dtype=bool)
        selected = []
        degrees = conflicts.sum(axis=1).astype(np.int32)
        rng = np.random.default_rng(42)
        
        while np.any(alive) and time.time() < deadline:
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
        
        result_vectors = vectors[selected]
    else:
        result_vectors = vectors
    
    return result_vectors
