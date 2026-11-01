"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # 1. Generate Type A vectors (Weight 1, value +/-2, Norm^2=4)
    type_a_vectors = []
    for i in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * sign
            type_a_vectors.append(v)
    
    type_a_array = np.array(type_a_vectors)
    n_a = len(type_a_array) # 22

    # 2. Generate Type C vectors (Weight 4, values +/-1, Norm^2=4)
    type_c_vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            type_c_vectors.append(v)
            
    type_c_array = np.array(type_c_vectors)
    n_c = len(type_c_array) # 5280

    # 3. Compute Conflict Graph for Type C vectors
    # Conflict if dot > 2
    dots = type_c_array @ type_c_array.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as lists of indices for faster iteration if needed, 
    # but boolean array is good for vectorized ops.
    # For Min-Degree Greedy, we need to update degrees.
    # neighbors[i] is a list of j such that conflicts[i,j] is True.
    neighbors = [np.where(conflicts[i])[0] for i in range(n_c)]
    degrees = np.array([len(nbrs) for nbrs in neighbors])

    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(n_c, dtype=bool)
    
    def greedy_min_degree():
        """
        Greedy MIS using dynamic minimum degree heuristic.
        """
        in_set = np.zeros(n_c, dtype=bool)
        available = np.ones(n_c, dtype=bool)
        
        # Priority queue: (degree, node_index)
        # We need to update degrees as we remove nodes.
        # Initial degrees in the full graph.
        # As we select a node u, we remove u and its neighbors from 'available'.
        # The degree of a remaining node v decreases by 1 for each neighbor of u that is also a neighbor of v?
        # No, the degree of v in the *induced subgraph on available* is:
        # count of w in available such that conflicts[v, w] is True.
        
        # This dynamic update is tricky with a simple heap because degrees change.
        # Lazy deletion in heap:
        # Push (current_degree, node).
        # When popping, check if node is still available and if its degree matches the heap entry.
        # If not, discard and push new degree.
        
        # To get current degree of v:
        # deg(v) = sum(available[neighbors[v]])
        
        # This is O(degree) per query.
        
        heap = []
        for i in range(n_c):
            heapq.heappush(heap, (degrees[i], i))
            
        current_size = 0
        
        while any(available):
            # Find min degree node in available
            u = -1
            while heap:
                d, i = heapq.heappop(heap)
                if not available[i]:
                    continue
                # Verify degree
                # If the degree changed, we need to re-push.
                # But checking exact degree is expensive.
                # Heuristic: Just use the stored degree. It might be stale (too high).
                # If we pick a node with stale high degree, it's suboptimal but valid.
                # To be better, we can re-calculate if we suspect staleness.
                # For now, let's assume the initial degree is a good proxy, 
                # or we update it.
                
                # Actually, let's just re-calculate degree for the top few candidates?
                # Or simpler: Use a static min-degree heuristic (pick node with min initial degree among available).
                # This is O(N) per step, O(N^2) total. 5000^2 = 25M. Fast enough.
                
                # Let's do static min-degree for simplicity and speed.
                # Find min degree among available nodes.
                break
            
            # Better approach for speed:
            # Just iterate through available nodes and pick the one with min degree.
            # We can vectorize this?
            # degs = np.array([np.sum(available[neighbors[i]]) for i in range(n_c)]) is slow.
            
            # Let's stick to the heap with lazy deletion, but re-calculate degree when popping.
            # If calculated degree != popped degree, push (calculated_degree, i) and continue.
            
            pass # Placeholder for logic below

    # Revised Greedy Min-Degree
    def greedy_min_degree_impl():
        in_set = np.zeros(n_c, dtype=bool)
        available = np.ones(n_c, dtype=bool)
        
        heap = []
        for i in range(n_c):
            heapq.heappush(heap, (degrees[i], i))
            
        current_size = 0
        
        while any(available):
            u = -1
            while heap:
                d_stale, i = heapq.heappop(heap)
                if not available[i]:
                    continue
                # Calculate true degree in available subgraph
                d_true = np.sum(available[neighbors[i]])
                if d_true != d_stale:
                    heapq.heappush(heap, (d_true, i))
                    continue
                u = i
                break
            
            if u == -1:
                break
                
            # Select u
            in_set[u] = True
            current_size += 1
            
            # Remove u and neighbors from available
            available[u] = False
            for nbr in neighbors[u]:
                if available[nbr]:
                    available[nbr] = False
                    # Note: We don't update heap entries for nbrs immediately.
                    # They will be detected as stale when popped.
                    
        return in_set, current_size

    # Local Search: 1-for-1 swaps
    def local_search(in_set, current_size):
        # Conflicts count for nodes NOT in set?
        # conf_counts[v] = number of neighbors of v that are in in_set.
        # v is free if conf_counts[v] == 0.
        
        conf_counts = np.zeros(n_c, dtype=np.int32)
        # Initialize conf_counts
        # For each u in in_set, increment conf_counts[neighbors[u]]
        for u in np.where(in_set)[0]:
            conf_counts[neighbors[u]] += 1
            
        # Track blocked nodes
        # A node v is blocked if conf_counts[v] > 0.
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any free node
            free_nodes = np.where((~in_set) & (conf_counts == 0))[0]
            if len(free_nodes) > 0:
                v = free_nodes[0] # Pick first
                in_set[v] = True
                conf_counts[neighbors[v]] += 1
                current_size += 1
                improved = True
                continue
            
            # 2. Try 1-for-1 swap
            # For each u in in_set, remove u, check if any node becomes free.
            # If yes, add it.
            # To speed up, we can check which nodes become free when u is removed.
            # Node v becomes free if conf_counts[v] == 1 and the only neighbor in set is u.
            # i.e., conf_counts[v] == 1 and conflicts[v, u] is True.
            
            # Precompute for each u, the list of v such that conf_counts[v]==1 and u is the blocker.
            # This is dynamic.
            
            # Simple loop:
            in_set_indices = np.where(in_set)[0]
            for u in in_set_indices:
                # Simulate removal of u
                # Nodes that might become free: neighbors of u that are NOT in set.
                # Check if their conf_counts drops to 0.
                # conf_counts[v] - 1 == 0 => conf_counts[v] == 1.
                
                # Candidates: neighbors[u] that are not in_set and have conf_counts == 1.
                nbrs = neighbors[u]
                candidates = nbrs[(~in_set[nbrs]) & (conf_counts[nbrs] == 1)]
                
                if len(candidates) == 0:
                    continue
                
                # Check if any candidate is free after removal.
                # Since we assume conf_counts is accurate, if conf_counts[v]==1 and u is in set,
                # removing u makes conf_counts[v]==0.
                # So any candidate is valid.
                # Pick one.
                v = candidates[0]
                
                # Perform swap
                in_set[u] = False
                conf_counts[neighbors[u]] -= 1
                
                in_set[v] = True
                conf_counts[neighbors[v]] += 1
                
                # Size unchanged, but configuration changed.
                # This might allow future additions.
                improved = True
                break # Restart loop to check for additions again
                
        return in_set, current_size

    # Run multiple restarts
    n_restarts = 0
    while time.time() < deadline - 5:
        n_restarts += 1
        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break
            
        in_set, size = greedy_min_degree_impl()
        in_set, size = local_search(in_set, size)
        
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()

    # Construct final result
    final_c_indices = np.where(best_in_set)[0]
    result_c = type_c_array[final_c_indices]
    
    # Combine Type A and Type C
    if len(result_c) > 0:
        final_vectors = np.vstack([type_a_array, result_c])
    else:
        final_vectors = type_a_array
        
    return final_vectors

# EVOLVE-BLOCK-END
