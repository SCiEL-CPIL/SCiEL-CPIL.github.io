import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # 1. Generate W4 vectors
    # Support size 4, entries +/-1
    w4_vectors = []
    supports = list(itertools.combinations(range(D), 4))
    for sup in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(sup, signs):
                v[pos] = s
            w4_vectors.append(v)
            
    w4_vectors = np.array(w4_vectors, dtype=np.int64)
    N4 = len(w4_vectors)
    
    # 2. Generate W1 vectors
    # Norm squared 4
    w1_vectors = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            w1_vectors.append(v)
            
    w1_vectors = np.array(w1_vectors, dtype=np.int64)

    # 3. Find large independent set in W4 conflict graph
    # Conflict if dot > 2
    # We can compute conflicts on the fly or precompute.
    # Precomputing 5280x5280 dot products is ~28M ops, fast.
    dots = w4_vectors @ w4_vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Greedy Min-Degree
    in_set = np.zeros(N4, dtype=bool)
    alive = np.ones(N4, dtype=bool)
    
    # Degree of each node
    degrees = conflicts.sum(axis=1)
    
    while np.any(alive) and time.time() < deadline - 5:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[0] # Deterministic for reproducibility, or random
        
        in_set[chosen] = True
        
        # Remove chosen and its neighbors from alive
        nbrs = np.where(conflicts[chosen] & alive)[0]
        to_remove = np.unique(np.concatenate(([chosen], nbrs)))
        alive[to_remove] = False
        
        # Update degrees
        # Degrees of remaining alive nodes decrease by number of removed neighbors
        rem = np.where(alive)[0]
        if len(rem) > 0:
             # conflicts[rem][:, to_remove] gives matrix of conflicts between rem and to_remove
             # Sum along axis 1 gives count of removed neighbors for each rem node
             degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
             
    # 4. Combine W4 independent set with all W1 vectors
    final_w4 = w4_vectors[in_set]
    
    # Verify W1 are compatible with W4 and each other (they are)
    # Result
    if len(final_w4) > 0:
        result = np.vstack([final_w4, w1_vectors])
    else:
        result = w1_vectors
        
    return result
