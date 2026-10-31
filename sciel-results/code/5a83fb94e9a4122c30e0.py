import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
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
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    # Compute conflict graph for V4
    # Conflict if dot product > 2
    # dots = vectors_v4 @ vectors_v4.T
    # This is 5280x5280 matrix. ~28M entries. Fits in memory.
    # 5280^2 * 8 bytes = ~224 MB. Might be tight but okay.
    # Let's try to compute it in chunks or use sparse if possible.
    # But for greedy MIS, we need fast neighbor lookups.
    # Dense matrix is faster for small N.
    
    if N4 == 0:
        return vectors_v1
    
    # Compute dot products
    # To save memory and time, we can compute conflicts on the fly or use a bitset.
    # But 5280 is small enough for dense matrix on modern machines.
    dots = vectors_v4 @ vectors_v4.T
    
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Greedy MIS
    alive = np.ones(N4, dtype=bool)
    selected_indices = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    rng = np.random.default_rng(42)
    
    # Optimization: Use a heap or sorted list for degrees?
    # Simple loop with min-degree selection.
    # N=5280. Loop runs ~500-1000 times (size of MIS).
    # Each step: find min degree among alive, remove neighbors.
    # Finding min degree: O(N).
    # Removing neighbors: O(N) to update alive, O(N^2) to update degrees?
    # degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
    # This is vectorized, so it's fast.
    
    while np.any(alive) and time.time() < deadline:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        # Find min degree
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected_indices.append(chosen)
        
        # Remove chosen and its neighbors
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        alive[to_remove] = False
        
        # Update degrees
        rem = np.where(alive)[0]
        if len(rem) > 0:
            # conflicts[rem][:, to_remove] is (len(rem), len(to_remove))
            # Sum along axis 1 gives number of removed neighbors for each remaining node
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        
        if time.time() > deadline - 10:
            break

    result_v4 = vectors_v4[selected_indices]
    
    # Combine V4 and V1
    # V1 vectors are all compatible with V4.
    # V1 vectors are compatible with each other (max 2 per dim).
    # We generated all 22 V1 vectors.
    # Check if any V1 conflicts with another V1?
    # We established we can take all 22.
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
