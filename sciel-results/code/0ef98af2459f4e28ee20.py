import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate all weight-4 vectors
    indices = range(D)
    pool = []
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool.append(v)
            
    pool = np.array(pool, dtype=np.int64)
    n_pool = len(pool)
    
    # Precompute the conflict graph (adjacency matrix or sparse representation)
    # Two vectors conflict if their dot product > 2 (since norm is 2, dist^2 = 8 - 2*dot < 4 => dot > 2)
    # Actually, condition is min pairwise distance >= max norm.
    # Norm is 2. Distance squared is ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 8 - 2 u.v.
    # We need 8 - 2 u.v >= 4  =>  4 >= 2 u.v  =>  u.v <= 2.
    # So conflict if u.v > 2.
    
    # Compute all pairwise dot products
    dots = pool @ pool.T
    # Conflict if dot > 2
    conflict_mask = dots > 2
    
    # We want to find a large independent set.
    # Degree of a vertex is the number of conflicts.
    # Strategy: Repeatedly pick the vertex with the lowest degree in the *remaining* graph.
    # This is the "Minimum Degree" heuristic for MIS.
    
    best_set = np.array([], dtype=np.int64).reshape(0, D)
    best_len = 0
    
    # We will run this heuristic multiple times with different random tie-breaks
    # to find a larger independent set.
    
    # Precompute degrees initially? No, degrees change as we remove vertices.
    # To speed up, we can maintain a list of active vertices and a matrix of active conflicts.
    # But n=5280, so 5280x5280 matrix is ~28M bools, which is fine for memory.
    # Operations on 5280x5280 matrices are fast in NumPy.
    
    # Let's implement a single pass of Min-Degree Greedy.
    # We can randomize the initial order or tie-breaking.
    
    for restart in range(20): # Try a few times
        if time.time() > deadline:
            break
            
        # Reset active mask
        active = np.ones(n_pool, dtype=bool)
        selected_indices = []
        
        # To make it randomized, we can add small random noise to degrees when ties occur.
        # Or simply shuffle the initial vertex order and break ties by index in that order.
        
        # Efficient Min-Degree Greedy:
        # 1. Calculate degrees of all active vertices.
        # 2. Pick the one with min degree (random tie-break).
        # 3. Add to selected.
        # 4. Remove it and all its neighbors from active.
        
        # We can optimize by precomputing the neighbor lists?
        # Or just use matrix operations.
        
        # Let's use a list of active indices to iterate over.
        active_indices = np.arange(n_pool)
        
        while len(active_indices) > 0:
            if time.time() > deadline:
                return best_set if best_len > 0 else np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
                
            # Calculate degrees for active vertices
            # degree[i] = sum of conflict_mask[i, active_indices]
            # This is O(N_active * N_total) if done naively.
            # Better: degrees = conflict_mask[active_indices, :].sum(axis=1)
            # But we only care about active neighbors.
            # degrees = (conflict_mask[np.ix_(active_indices, active_indices)]).sum(axis=1)
            # This creates a submatrix. For 5280, it's okay for a few iterations, but might be slow for 500 iterations.
            
            # Alternative: Maintain a degree array and update it.
            # This is complex to implement correctly with random removals.
            
            # Let's try the simple matrix sub-indexing first. If it's too slow, we'll optimize.
            # 5000 iterations * (5000x5000 sum) might be too slow.
            # 5000 * 25M ops = 125B ops. Too slow.
            
            # We need a faster degree calculation.
            # Let's maintain a `degree` array.
            # When we select vertex v, we subtract 1 from the degree of all its active neighbors.
            
            # Re-implement with dynamic degree tracking.
            # This requires restarting the inner loop logic.
            
            # Let's break out and implement the optimized version below.
            break 
            
        # The simple loop above is too slow for the while loop. 
        # Let's implement the optimized Min-Degree Greedy.
        
        active = np.ones(n_pool, dtype=bool)
        degrees = conflict_mask.sum(axis=1).astype(np.int32)
        selected_indices = []
        
        # We need to pick min degree among active.
        # To handle ties randomly, we can add a small random epsilon or just pick randomly from the min-degree set.
        
        # Optimization: Instead of scanning all active vertices every time, 
        # we can use a heap or just scan. Scanning 5280 elements is fast (~microseconds).
        # The bottleneck is updating degrees.
        # When we pick v, we need to update degrees of all active neighbors of v.
        # neighbors = np.where(conflict_mask[v] & active)[0]
        # degrees[neighbors] -= 1
        # active[v] = False
        
        # This is efficient.
        
        while True:
            if time.time() > deadline:
                break
                
            # Find active vertices
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            # Find min degree among active
            # We can just look at degrees[active_indices]
            min_deg = np.min(degrees[active_indices])
            
            # Find all active vertices with this min degree
            candidates = active_indices[degrees[active_indices] == min_deg]
            
            # Pick one randomly
            v = np.random.choice(candidates)
            
            selected_indices.append(v)
            
            # Update active and degrees
            # Find neighbors of v that are active
            # conflict_mask[v] is a bool array of length n_pool
            neighbors = np.where(conflict_mask[v] & active)[0]
            
            # Decrease degree of neighbors
            degrees[neighbors] -= 1
            
            # Mark v and its neighbors as inactive
            active[v] = False
            active[neighbors] = False
            
        # End of one greedy pass
        
        current_len = len(selected_indices)
        if current_len > best_len:
            best_len = current_len
            best_set = pool[selected_indices]
            
    return best_set
# EVOLVE-BLOCK-END
