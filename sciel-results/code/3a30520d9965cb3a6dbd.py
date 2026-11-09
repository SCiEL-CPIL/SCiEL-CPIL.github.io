import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a K2,3-free graph on 20 vertices maximizing edges.
    
    Strategy:
    1. Use a NumPy adjacency matrix for fast vectorized operations.
    2. Start with multiple random initial graphs (including the 5xK4 structure).
    3. Use a greedy local search that attempts to add edges.
       - For each potential edge (u,v), check if adding it violates the K2,3 condition.
       - The condition is: No two vertices have 3 common neighbors.
       - When adding edge (u,v):
         - Let C_uv = N(u) ∩ N(v). If |C_uv| >= 3, invalid.
         - For any w in N(u) \ {v}, u becomes a common neighbor of w and v.
           So we need |N(w) ∩ N(v)| <= 1.
         - Symmetrically for w in N(v) \ {u}, |N(w) ∩ N(u)| <= 1.
    4. Keep the best graph found across all runs within the time limit.
    """
    n = 20
    start_time = time.time()
    time_limit = 140.0  # Safety margin below 150s limit
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 1:
                    edges.append([u, v])
        return edges

    def count_edges(adj):
        return int(np.sum(adj)) // 2

    def is_k23_free(adj):
        """
        Check if the graph is K2,3-free.
        K2,3 exists if there are two vertices u,v with >=3 common neighbors.
        """
        # Compute common neighbors for all pairs
        # adj @ adj gives counts of common neighbors for all pairs (u,v)
        # Note: adj[u,v] is 1 if edge exists.
        # common[u,v] = sum_k adj[u,k]*adj[v,k]
        common = adj @ adj
        # We need to check if any common[u,v] >= 3
        # Note: The diagonal common[u,u] is degree(u), which can be > 2.
        # We only care about off-diagonal pairs (u != v).
        # Also, if u and v are not connected, they can still have common neighbors.
        # The definition of K2,3 subgraph usually implies the two vertices in the '2' part 
        # don't need to be connected to each other, but they share 3 neighbors.
        # Wait, K2,3 is a bipartite graph. The vertices in the partition of size 2 
        # do NOT have an edge between them in the induced subgraph of K2,3?
        # Actually, K2,3 is just a set of 5 vertices {u,v} U {w1,w2,w3} 
        # such that u is connected to w1,w2,w3 and v is connected to w1,w2,w3.
        # The edge (u,v) is irrelevant to the existence of the K2,3 subgraph 
        # unless we are looking for induced subgraphs or specific definitions.
        # The standard extremal number ex(n, K2,3) forbids any subgraph isomorphic to K2,3.
        # So if u and v have 3 common neighbors, a K2,3 exists regardless of whether (u,v) is an edge.
        
        # So we check: does there exist u != v such that common[u,v] >= 3?
        # Mask out diagonal
        np.fill_diagonal(common, 0)
        if np.any(common >= 3):
            return False
        return True

    def try_add_edges_greedy(adj, max_iterations=1000):
        """
        Greedily add edges while maintaining K2,3-freeness.
        Returns the modified adj matrix and edge count.
        """
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try all non-edges
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v] == 1:
                        continue
                    
                    # Check if adding (u,v) is valid
                    # 1. Common neighbors of u and v must be <= 2
                    common_uv = np.sum(adj[u, :] & adj[v, :])
                    if common_uv > 2:
                        continue
                    
                    # 2. For all w in N(u), common neighbors of w and v must be <= 1
                    #    Because u will become a common neighbor of w and v.
                    #    So we need |N(w) ∩ N(v)| <= 1 for all w in N(u).
                    #    Note: v is not in N(u) yet.
                    
                    # Get neighbors of u
                    neighbors_u = np.where(adj[u, :] == 1)[0]
                    valid = True
                    for w in neighbors_u:
                        # Common neighbors of w and v
                        common_wv = np.sum(adj[w, :] & adj[v, :])
                        if common_wv > 1:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # 3. Symmetric check: For all w in N(v), common neighbors of w and u must be <= 1
                    neighbors_v = np.where(adj[v, :] == 1)[0]
                    for w in neighbors_v:
                        common_wu = np.sum(adj[w, :] & adj[u, :])
                        if common_wu > 1:
                            valid = False
                            break
                    
                    if valid:
                        adj[u, v] = 1
                        adj[v, u] = 1
                        improved = True
        
        return adj

    def random_initial_graph(density=0.3):
        adj = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if np.random.rand() < density:
                    adj[u, v] = 1
                    adj[v, u] = 1
        return adj

    def k4_union_graph():
        adj = np.zeros((n, n), dtype=int)
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    adj[u, v] = 1
                    adj[v, u] = 1
        return adj

    best_edges = 0
    best_adj = np.zeros((n, n), dtype=int)
    
    # Initial candidates
    candidates = [k4_union_graph()]
    
    # Add some random starts
    for _ in range(10):
        candidates.append(random_initial_graph(density=0.2))
        candidates.append(random_initial_graph(density=0.4))
        
    # Shuffle candidates
    np.random.shuffle(candidates)
    
    for i, adj_init in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
            
        # Make a copy
        adj = adj_init.copy()
        
        # Ensure it's K2,3 free initially? 
        # If not, we might need to remove edges first. 
        # But greedy addition only adds edges. 
        # If the initial graph is invalid, adding more edges won't fix it.
        # So we must ensure the starting graph is valid.
        if not is_k23_free(adj):
            # Try to repair by removing edges? Or just skip?
            # Let's try a simple repair: remove edges that cause K2,3.
            # This is expensive. For now, skip invalid starts.
            # Better: Generate valid random graphs.
            # A random graph with low density is likely valid.
            # Let's just skip if invalid to be safe, or use a very low density.
            continue
        
        # Run greedy local search
        adj = try_add_edges_greedy(adj)
        
        # Verify validity (should be true if logic is correct)
        if is_k23_free(adj):
            edges_count = count_edges(adj)
            if edges_count > best_edges:
                best_edges = edges_count
                best_adj = adj.copy()
        
        # Optional: Try some random swaps or removals to escape local optima?
        # For now, just stick to greedy addition from multiple starts.
        
    # Final check and extraction
    if best_edges == 0:
        # Fallback to K4 union if something went wrong
        adj = k4_union_graph()
        best_adj = adj
        best_edges = count_edges(adj)

    edges = get_edges_from_adj(best_adj)
    return edges
# EVOLVE-BLOCK-END
