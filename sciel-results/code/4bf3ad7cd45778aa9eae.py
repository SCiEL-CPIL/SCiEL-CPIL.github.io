import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin for 150s limit
    
    # 1. Generate Candidates
    # Norm squared 2: vectors with two 1s/-1s
    # Norm squared 3: vectors with three 1s/-1s
    
    candidates = []
    norms_sq = []
    
    # Generate weight 2 vectors
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)
                norms_sq.append(2)
                
    # Generate weight 3 vectors
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
                    norms_sq.append(3)
                    
    n = len(candidates)
    V = np.array(candidates, dtype=np.int8)
    
    # 2. Compute Conflict Graph
    # Conflict if dot(u, v) >= 2
    # We can compute all pairwise dot products
    # V is (n, D). Dot products are V @ V.T
    # This is n x n. n=1540. 1540^2 ~ 2.4M. Fast.
    
    dot_prods = V @ V.T
    
    # Conflict matrix: 1 if dot >= 2
    # Note: diagonal is norm_sq (2 or 3), which is >= 2, so self-conflict is true.
    # We want an independent set, so we ignore self-loops in selection logic 
    # but the mask should be True for conflicts between distinct nodes.
    
    conflict = dot_prods >= 2
    
    # 3. Greedy Maximum Independent Set
    # Heuristic: Pick node with lowest degree in remaining graph.
    # To speed up, we can sort by initial degree and pick from lowest available.
    
    degrees = conflict.sum(axis=1)
    # We maintain a set of available nodes
    available = np.ones(n, dtype=bool)
    selected = []
    
    # To make it faster, we can pre-sort indices by degree
    # But degrees change as we remove nodes.
    # Simple approach: 
    # Loop n times. Find min degree among available. Pick it. Remove it and neighbors.
    # This is O(n^3) worst case if done naively with full scans.
    # n=1540. n^3 is 3.6e9. Too slow in Python.
    
    # Better Greedy:
    # 1. Sort nodes by initial degree.
    # 2. Iterate through sorted nodes.
    # 3. If node is still available, select it, mark it and its neighbors as unavailable.
    # This is O(n^2) roughly.
    
    order = np.argsort(degrees)
    
    for idx in order:
        if available[idx]:
            selected.append(idx)
            # Remove neighbors
            # conflict[idx] is a boolean mask of neighbors (and self)
            available[conflict[idx]] = False
            
    current_set = np.array(selected)
    current_size = len(current_set)
    
    # 4. Local Search (Iterated Local Search)
    # Try to improve by removing a small random subset and re-greedy-ing on the freed space + removed set.
    
    best_set = current_set.copy()
    best_size = current_size
    
    # Helper: Greedy fill on a subset of candidates given a "blocked" mask
    # blocked[i] is True if candidate i cannot be added.
    # We want to add nodes from 'allowed' to 'current_selection'.
    
    def greedy_fill(allowed_mask, current_selection_indices, conflict_matrix, n_nodes):
        # current_selection_indices: list of indices already selected
        # We want to pick from allowed_mask.
        # A node can be picked if it is in allowed_mask AND not conflicting with current_selection
        # AND not conflicting with previously picked nodes in this step.
        
        # Create a temporary available mask
        temp_available = allowed_mask.copy()
        
        # Mask out nodes conflicting with existing selection
        if len(current_selection_indices) > 0:
            # conflict_matrix[current_selection_indices] is (k, n)
            # Any node in temp_available that is True in any of these rows must be removed
            # Use any() along axis 0
            conflicts_with_existing = conflict_matrix[current_selection_indices].any(axis=0)
            temp_available[conflicts_with_existing] = False
            
        new_selected = []
        # Sort temp_available indices by degree in the *original* graph? 
        # Or just iterate? 
        # For speed, just iterate through a pre-sorted list of all nodes and check if available.
        # But we need to update availability as we pick.
        
        # Efficient way:
        # While there are available nodes:
        #   Pick one (e.g., lowest degree in original graph, or random)
        #   Remove it and its neighbors from temp_available
        
        # Let's use the pre-sorted 'order' list
        for idx in order:
            if temp_available[idx]:
                new_selected.append(idx)
                temp_available[conflict[idx]] = False
                
        return new_selected

    # Local Search Loop
    t_start_ls = time.time()
    while time.time() - t_start_ls < 100: # Run for ~100s
        
        # Randomly remove 5-10 nodes from best_set
        k_remove = min(10, best_size)
        if k_remove == 0: break
        
        remove_indices = np.random.choice(best_size, k_remove, replace=False)
        
        # Create mask of nodes to "free up"
        # We will try to re-select from: 
        # 1. The removed nodes
        # 2. Any node NOT in best_set that is not conflicting with (best_set - removed)
        
        remaining_best = np.delete(best_set, remove_indices)
        
        # Identify candidates for the "hole"
        # Candidates are:
        # - The removed nodes
        # - All other nodes not in remaining_best
        
        # Create a boolean mask for "allowed" candidates
        # allowed[i] = True if i is not in remaining_best
        allowed_mask = np.ones(n, dtype=bool)
        allowed_mask[remaining_best] = False
        
        # Run greedy fill on allowed_mask, starting with remaining_best
        new_additions = greedy_fill(allowed_mask, remaining_best, conflict, n)
        
        new_set = np.concatenate([remaining_best, new_additions])
        
        if len(new_set) > best_size:
            best_size = len(new_set)
            best_set = new_set
            # Restart timer? No, just continue.
            # Maybe reset the local search timer if we found a big jump?
            # For now, keep going.
            
    # Return the best set found
    return V[best_set]

# EVOLVE-BLOCK-END
