import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # 1. Generate Norm 2 vectors (Baseline: 220 points)
    # These always form a valid set.
    candidates_2 = []
    for combo in itertools.combinations(range(d), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_2.append(v)
    
    base_solution = np.array(candidates_2, dtype=np.int64)
    best_points = base_solution
    best_k = len(base_solution)
    
    # 2. Generate Norm 4 vectors
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_4.append(v)
            
    if not candidates_4:
        return best_points
        
    V4 = np.array(candidates_4, dtype=np.int64)
    N4 = len(V4)
    
    # 3. Build Conflict Graph for Norm 4 vectors
    # Condition: dot product > 2 means conflict (distance < R=4)
    # We want Max Independent Set (no conflicts).
    # Compute all dot products in chunks to save memory and allow time checks.
    
    # We store adjacency as a list of numpy arrays for fast boolean masking in the greedy step.
    conflict_adj = [None] * N4
    
    # Chunk size 2000 seems reasonable for memory (2000 * 15840 * 8 bytes ~ 250MB max transient, but actually we compute row by row or small chunks)
    # Let's use a smaller chunk to be safe on memory, say 500.
    # Or better, just compute row by row? 15840 dot products of length 11 is very fast.
    # V4[i] @ V4.T is (1, 15840). This is very fast in numpy.
    # Let's try row-by-row to minimize memory overhead and allow fine-grained time checking.
    
    for i in range(N4):
        if time.time() > deadline - 10:
            # If we run out of time, we can't build the full graph.
            # Return best found so far.
            return best_points
        
        dots_row = V4[i] @ V4.T
        # Conflict if dot > 2
        # We exclude self (i) as it's not a pair in the independent set logic, 
        # but for the greedy "remove neighbors" step, we just remove all conflicting nodes.
        # Self is already removed from active, so it doesn't matter if it's in the list.
        # However, to save memory/speed, we can exclude i.
        neighbors = np.where(dots_row > 2)[0]
        # Exclude self if present (it will be, since dot(V4[i], V4[i]) = 4 > 2)
        if i in neighbors:
             # This check is O(1) average if we use a set, but neighbors is an array.
             # Simpler: just keep it. When we do active_mask[neighbors] = False,
             # it sets active_mask[i] to False, which is already False. No harm.
             pass
        conflict_adj[i] = neighbors

    # 4. Greedy Search with Random Restarts
    # Use numpy-based active set management for speed
    best_indices = []
    
    while time.time() < deadline - 5:
        order = np.random.permutation(N4)
        
        # active_mask: boolean array, True if vertex is still available
        active_mask = np.ones(N4, dtype=bool)
        current_indices = []
        
        for v in order:
            if not active_mask[v]:
                continue
            current_indices.append(v)
            # Remove v and all its neighbors from active
            active_mask[v] = False
            if len(conflict_adj[v]) > 0:
                active_mask[conflict_adj[v]] = False
                
        if len(current_indices) > len(best_indices):
            best_indices = current_indices
            if len(best_indices) > best_k:
                new_points = V4[best_indices].astype(np.int64)
                best_points = new_points
                best_k = len(best_indices)
        
        # Early termination if we hit a very high score?
        # The theoretical max for norm 4 vectors might be around 300-400?
        # If we get close to that, we can stop.
        # But let's just run until deadline.
        
    return best_points
