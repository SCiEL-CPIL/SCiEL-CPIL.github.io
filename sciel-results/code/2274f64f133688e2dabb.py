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
    
    # Compute conflicts
    # Dot product > 2 is a conflict
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Pre-compute adjacency lists as lists of indices for faster access?
    # Or stick to matrix for vectorization.
    
    best_selected = np.zeros(N4, dtype=bool)
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple greedy MIS attempts with incremental degree updates
    max_restarts = 200 # Can increase if fast
    
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        alive = np.ones(N4, dtype=bool)
        selected_indices = []
        # Initial degrees
        degrees = conflicts.sum(axis=1).astype(np.int32).copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            # Find min degree among alive
            # Mask out non-alive
            degs_masked = np.where(alive, degrees, np.inf)
            min_idx = np.argmin(degs_masked)
            
            if not alive[min_idx]:
                break
                
            selected_indices.append(min_idx)
            
            # Get neighbors
            neighbors = np.where(conflicts[min_idx])[0]
            to_remove = np.unique(np.concatenate(([min_idx], neighbors)))
            
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Subtract 1 for each neighbor that was removed
                # adj[rem, to_remove] is boolean
                # Sum along axis 1
                degrees[rem] -= (conflicts[rem][:, to_remove] == 1).sum(axis=1)
                
        # Local Search: Try to add back nodes?
        # Simple: Check if any unselected node has no neighbors in selected.
        # This is usually already handled by the greedy process if it terminates correctly.
        # But due to time limits or greedy choices, we might miss some.
        
        # More powerful local search:
        # Try to swap: remove u, add v?
        # For now, just count.
        
        count = len(selected_indices)
        if count > best_count:
            best_count = count
            best_selected = np.zeros(N4, dtype=bool)
            best_selected[selected_indices] = True

    result_v4 = vectors_v4[best_selected]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
