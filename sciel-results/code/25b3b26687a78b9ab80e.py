import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 112

    # Generate W1 vectors (norm 2)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Generate W4 vectors (norm 2)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Compute Gram matrix for W4 vectors
    # G[i, j] = dot(v4[i], v4[j])
    G = vectors_v4 @ vectors_v4.T
    
    # Conflict if G[i, j] > 2
    # Diagonal is 4, so we ignore i==j
    conflicts = G > 2
    np.fill_diagonal(conflicts, False)
    
    # We want Max Independent Set in this graph
    # Initialize with empty set
    best_set = np.zeros(N4, dtype=bool)
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    def greedy_mis(conflict_graph, rng):
        n = len(conflict_graph)
        in_set = np.zeros(n, dtype=bool)
        alive = np.ones(n, dtype=bool)
        
        # Precompute degrees
        degrees = conflict_graph.sum(axis=1)
        
        while np.any(alive):
            # Find node with min degree among alive
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            # Remove chosen and its neighbors from alive
            nbrs = np.where(conflict_graph[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            # Degree decreases by number of removed neighbors
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Count how many of the removed nodes were neighbors of rem
                # conflict_graph[rem, to_remove] is a boolean matrix
                # Sum along axis 1 gives count of removed neighbors for each rem node
                degrees[rem] -= conflict_graph[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search_mis(in_set_init, conflict_graph, max_iters=1000):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # Precompute conflict counts for nodes not in set
        # conflict_count[i] = number of neighbors of i that are in in_set
        conflict_count = np.zeros(len(in_set), dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[conflict_graph[i]] += 1
            
        stuck = 0
        for _ in range(max_iters):
            # Find a node not in set with 0 conflicts (can be added)
            free = np.where((~in_set) & (conflict_count == 0))[0]
            if len(free) > 0:
                # Add all free nodes? Or one by one?
                # Adding one might block others.
                # Greedy: add one with min degree in the "free" graph?
                # Just add one random free node
                chosen = free[rng.integers(len(free))]
                in_set[chosen] = True
                current_size += 1
                # Update conflict counts for neighbors of chosen
                nbrs = np.where(conflict_graph[chosen])[0]
                conflict_count[nbrs] += 1
                stuck = 0
                continue
                
            # If no free nodes, try a swap?
            # Find node not in set with min conflict count > 0
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
                
            min_conf = np.min(conflict_count[not_in_set])
            if min_conf == 0:
                # Should have been caught above
                continue
                
            # Try to find a swap that increases size
            # This is hard. Let's just restart or accept local optimum.
            # For now, break if stuck too long
            stuck += 1
            if stuck > 100:
                break
                
        return in_set, current_size

    # Run multiple restarts
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        in_set = greedy_mis(conflicts, rng)
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_set = in_set.copy()
            
        # Local search polish
        ls_set, ls_size = local_search_mis(best_set, conflicts, max_iters=500)
        if ls_size > best_size:
            best_size = ls_size
            best_set = ls_set.copy()

    # Combine W4 and W1
    selected_v4 = vectors_v4[best_set]
    final_vectors = np.vstack([selected_v4, vectors_v1])
    
    return final_vectors
