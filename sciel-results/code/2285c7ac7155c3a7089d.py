# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Improved greedy approach on candidate vectors with norm^2 = 2, 3, and 4.
    Includes norm^2=3 vectors (three ±1's) to increase candidate pool.
    Optimized greedy selection using degree tracking.
    """
    D = 11
    
    # Generate candidate vectors
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 2: three ±1's, rest 0 (norm^2 = 3)
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    # Type 3: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 4: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = np.max(norms_sq) # Should be 4
    
    # Compute pairwise dot products
    dots = candidates @ candidates.T
    
    # Compute squared distances
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict matrix: conflict[i,j] = True if dist_sq[i,j] < max_norm_sq
    # Note: We must ensure distance >= max_norm.
    # If max_norm_sq is 4, we need dist_sq >= 4.
    # So conflict if dist_sq < 4.
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    # Greedy selection: repeatedly pick a vector with fewest conflicts among remaining
    # To speed up, we maintain a list of active nodes and their degrees.
    active = np.ones(n, dtype=bool)
    degrees = conflict.sum(axis=1).astype(np.int32)
    selected = []
    
    # Use a heap or just find min each time?
    # Finding min in an array of size N is O(N).
    # Total O(N*k).
    # N ~ 7000, k ~ 500. 3.5M ops. Fast.
    
    np.random.seed(42)
    
    for _ in range(n): # Max iterations
        if not np.any(active):
            break
        
        # Find indices of active nodes
        active_indices = np.where(active)[0]
        
        # Get degrees of active nodes
        active_degrees = degrees[active_indices]
        
        # Find min degree
        min_deg = np.min(active_degrees)
        
        # Find all active nodes with min degree
        min_mask = (active_degrees == min_deg)
        min_candidates = active_indices[min_mask]
        
        # Randomly pick one
        chosen = np.random.choice(min_candidates)
        
        selected.append(chosen)
        
        # Update active and degrees
        # Remove chosen and its neighbors from active
        neighbors = np.where(conflict[chosen])[0]
        to_remove = np.concatenate(([chosen], neighbors))
        
        # Update degrees for remaining active nodes
        # For each node u in to_remove, for each neighbor v of u that is still active,
        # degree[v] -= 1
        
        # Vectorized update is tricky because of dynamic set.
        # Loop over to_remove nodes
        for u in to_remove:
            if not active[u]:
                continue
            active[u] = False
            # Get neighbors of u
            neigh_u = np.where(conflict[u])[0]
            # Only those that are still active
            still_active_neigh = neigh_u[active[neigh_u]]
            degrees[still_active_neigh] -= 1
            
    result = candidates[selected]
    return result
# EVOLVE-BLOCK-END
