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
    
    if N4 == 0:
        return vectors_v1

    # Compute dot products
    # 5280x5280 matrix of int64s is ~224MB. This is acceptable.
    # The operation is a single BLAS call, which is highly optimized.
    dots = vectors_v4 @ vectors_v4.T

    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Greedy MIS (Minimum Degree Heuristic)
    # We want to find a large subset of vectors with no conflicts (dot <= 2).
    # This is equivalent to finding a Maximum Independent Set in the conflict graph.
    # The Minimum Degree heuristic is a standard greedy approximation for MIS.
    
    alive = np.ones(N4, dtype=bool)
    selected_indices = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    rng = np.random.default_rng(42)

    while np.any(alive) and time.time() < deadline:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        # Find min degree among alive nodes
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected_indices.append(chosen)
        
        # Remove chosen and its neighbors
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        rem = np.where(alive)[0]
        if len(rem) > 0:
            # For each remaining node, subtract the number of its neighbors that were removed
            # conflicts[rem, to_remove] is a boolean matrix (len(rem) x len(to_remove))
            # Summing along axis 1 gives the count of removed neighbors for each remaining node
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        
        if time.time() > deadline - 5:
            break

    result_v4 = vectors_v4[selected_indices]
    
    # Combine V4 and V1
    # V1 vectors are all compatible with V4 (dot product is 0 or 2, never > 2).
    # V1 vectors are compatible with each other (dot product is 0 or 4? No.
    # v1 = (2,0,...), v1' = (-2,0,...) -> dot = -4? No, they are distinct vectors.
    # v1 = (2,0,...), v1' = (2,0,...) -> identical, not allowed.
    # v1 = (2,0,...), v1' = (0,2,...) -> dot = 0.
    # v1 = (2,0,...), v1' = (-2,0,...) -> dot = -4.
    # Wait, the condition is min distance >= max norm.
    # Norm^2 = 4 for all.
    # Distance^2 = ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j = 8 - 2 v_i.v_j.
    # We need 8 - 2 v_i.v_j >= 4 => 4 >= 2 v_i.v_j => v_i.v_j <= 2.
    # For V1 vectors:
    # If they are on the same axis and same sign: identical (excluded).
    # If they are on the same axis and opposite sign: dot = -4. -4 <= 2. OK.
    # If they are on different axes: dot = 0. 0 <= 2. OK.
    # So all 22 V1 vectors are mutually compatible.
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
