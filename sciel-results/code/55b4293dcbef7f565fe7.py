"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m_total = len(all_edges)
    
    # Map edge index to (u, v) for fast lookup
    edge_map = {}
    for i, (u, v) in enumerate(all_edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i

    # Initialize adjacency matrix as numpy array for vectorized operations
    # adj[u, v] = 1 if edge exists
    adj = np.zeros((n, n), dtype=np.uint8)
    
    # Helper to check validity of a graph represented by adjacency matrix
    # Returns True if no pair of vertices has > 2 common neighbors
    def is_valid(adj_mat):
        # Compute common neighbors for all pairs
        # C = adj @ adj.T gives counts of common neighbors for all pairs
        # Note: adj[i,j]=1 means edge. (adj @ adj.T)[i,j] = sum_k adj[i,k]*adj[k,j]
        # This counts paths of length 2.
        # For a simple graph, the number of common neighbors of i and j is (adj @ adj.T)[i,j].
        # We need this count to be <= 2 for all i != j.
        # Note: (adj @ adj.T)[i,i] is the degree of i, which we don't care about for K2,3 constraint 
        # (K2,3 requires two distinct vertices).
        
        common_counts = adj_mat @ adj_mat.T
        # Check off-diagonal elements
        # We can use np.fill_diagonal to ignore self-loops
        np.fill_diagonal(common_counts, 0)
        return np.max(common_counts) <= 2

    # Helper to get current edge count
    def get_edge_count(adj_mat):
        return int(np.sum(adj_mat)) // 2

    # Initialize with a random sparse graph to start SA
    # We want to start reasonably dense but valid.
    # Let's try a random graph with p=0.4 and then remove violations? 
    # Or just start empty and let SA add edges? 
    # Starting empty is safe but slow to build up.
    # Let's start with a random set of edges that is likely valid.
    # A random graph G(n, 0.5) on 20 nodes has ~95 edges. 
    # K2,3-free max edges is around 100-120? (Kozma-Szabo bound ~ n^2/4 + ... )
    # For n=20, n^2/4 = 100. So ~100 is a good target.
    
    best_adj = np.zeros((n, n), dtype=np.uint8)
    best_score = 0
    
    # Try multiple initializations
    num_inits = 5
    
    for init_idx in range(num_inits):
        if time.time() > deadline:
            break
            
        # Create a random initial graph
        curr_adj = np.zeros((n, n), dtype=np.uint8)
        
        # Randomly select a subset of edges
        # Start with a moderate density
        p = 0.3 + 0.1 * init_idx
        for u, v in all_edges:
            if random.random() < p:
                curr_adj[u, v] = 1
                curr_adj[v, u] = 1
        
        # Repair the initial graph to make it valid
        # Remove edges that cause violations
        improved = True
        while improved and time.time() < deadline:
            improved = False
            if not is_valid(curr_adj):
                # Find a violating pair
                common_counts = curr_adj @ curr_adj.T
                np.fill_diagonal(common_counts, 0)
                # Find index with max common neighbors
                max_count = np.max(common_counts)
                if max_count > 2:
                    # Find a pair (i, j) with max_count
                    idx = np.argmax(common_counts)
                    i = idx // n
                    j = idx % n
                    # Remove a random edge incident to i or j? 
                    # Better: remove an edge (i, k) or (j, k) where k is a common neighbor?
                    # Actually, removing any edge incident to i or j reduces the count for pairs involving i or j.
                    # Let's pick a random neighbor of i or j to remove the edge to.
                    neighbors_i = np.where(curr_adj[i, :])[0]
                    neighbors_j = np.where(curr_adj[j, :])[0]
                    # Combine and pick one
                    candidates = list(set(neighbors_i.tolist()) | set(neighbors_j.tolist()))
                    if candidates:
                        k = random.choice(candidates)
                        # Remove edge (i, k) or (j, k)? 
                        # Let's remove edge (i, k) if it exists
                        if curr_adj[i, k] == 1:
                            curr_adj[i, k] = 0
                            curr_adj[k, i] = 0
                        else:
                            curr_adj[j, k] = 0
                            curr_adj[k, j] = 0
                        improved = True
        
        curr_score = get_edge_count(curr_adj)
        if curr_score > best_score:
            best_score = curr_score
            best_adj = curr_adj.copy()
        
        # Simulated Annealing
        if time.time() > deadline:
            continue
            
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        max_iters = 5000 # Adjust based on speed
        
        curr_adj_sa = curr_adj.copy()
        curr_score_sa = curr_score
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random move: add or remove an edge
            u, v = random.choice(all_edges)
            
            # Check if move is valid in the new state
            # We need to check if adding/removing (u,v) keeps the graph K2,3-free.
            
            old_state = curr_adj_sa[u, v]
            
            # Tentatively apply move
            if old_state == 0:
                # Add edge
                curr_adj_sa[u, v] = 1
                curr_adj_sa[v, u] = 1
                new_score = curr_score_sa + 1
            else:
                # Remove edge
                curr_adj_sa[u, v] = 0
                curr_adj_sa[v, u] = 0
                new_score = curr_score_sa - 1
            
            # Check validity
            if is_valid(curr_adj_sa):
                # Accept if valid
                if new_score > curr_score_sa:
                    curr_score_sa = new_score
                else:
                    # Metropolis criterion for equal score (neutral moves) or worse?
                    # Since we maximize edges, we only want to accept if score increases or stays same.
                    # If score stays same, we might accept with prob exp(0/T) = 1? 
                    # Usually SA accepts worse states to escape local optima. 
                    # Here, "worse" means fewer edges. We want to avoid fewer edges unless it allows 
                    # future improvements. 
                    # Let's accept neutral moves with high probability to explore.
                    if random.random() < 0.5:
                        curr_score_sa = new_score
                    else:
                        # Revert
                        curr_adj_sa[u, v] = old_state
                        curr_adj_sa[v, u] = old_state
                        curr_score_sa = curr_score_sa # Revert score if reverted? 
                        # Wait, if we revert, score is back to original.
                        # Let's just track state.
                        pass # State is already reverted in adj, score logic needs care.
                        
                # Update best
                if curr_score_sa > best_score:
                    best_score = curr_score_sa
                    best_adj = curr_adj_sa.copy()
            else:
                # Invalid state, revert
                curr_adj_sa[u, v] = old_state
                curr_adj_sa[v, u] = old_state
                # Score remains curr_score_sa (original)
            
            T *= cooling
            if T < T_min:
                break

    # Final greedy fill on the best solution found
    if time.time() < deadline:
        final_adj = best_adj.copy()
        final_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any missing edge
            # To speed up, we can check candidates in random order
            candidates = [(u, v) for u, v in all_edges if final_adj[u, v] == 0]
            random.shuffle(candidates)
            for u, v in candidates:
                if time.time() > deadline:
                    break
                final_adj[u, v] = 1
                final_adj[v, u] = 1
                if is_valid(final_adj):
                    final_score += 1
                    improved = True
                else:
                    final_adj[u, v] = 0
                    final_adj[v, u] = 0
    
    # Convert back to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    return edges
# EVOLVE-BLOCK-END
