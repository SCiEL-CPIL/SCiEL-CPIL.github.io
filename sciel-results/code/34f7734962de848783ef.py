import numpy as np
import random

def construct():
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def greedy_construct(adj, edges_set):
        """
        Attempts to add more edges to the current graph (adj, edges_set) 
        using a greedy strategy until no more edges can be added.
        Returns the updated adj and edges_set.
        """
        # Create a list of candidate edges not in edges_set
        candidates = [e for e in all_pairs if e not in edges_set]
        random.shuffle(candidates)
        
        for u, v in candidates:
            # Check if adding (u, v) creates a 4-cycle
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            
            safe = True
            if len(nu) > 0 and len(nv) > 0:
                # Check submatrix adj[nu, nv] for any 1
                if np.any(adj[np.ix_(nu, nv)]):
                    safe = False
            
            if safe:
                edges_set.add((u, v))
                adj[u, v] = 1
                adj[v, u] = 1
                
        return adj, edges_set

    best_edges = 0
    best_graph = []
    
    # Initial greedy construction
    # We do a few initial runs to find a good starting point
    for _ in range(10):
        adj = np.zeros((n, n), dtype=np.int8)
        edges = set()
        adj, edges = greedy_construct(adj, edges)
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = list(edges)
            
    # Local Search: Remove an edge, then greedily refill
    # We work on the best graph found so far
    adj_best = np.zeros((n, n), dtype=np.int8)
    for u, v in best_graph:
        adj_best[u, v] = 1
        adj_best[v, u] = 1
    edges_best = set(best_graph)
    
    # Number of local search iterations
    # 120s limit. n=20 is small. Greedy is fast.
    # Let's do 500 iterations of "remove one, refill"
    for _ in range(500):
        # Copy current state
        adj_curr = adj_best.copy()
        edges_curr = set(edges_best)
        
        # Remove a random edge
        if not edges_curr:
            continue
        edge_to_remove = random.choice(list(edges_curr))
        u_rm, v_rm = edge_to_remove
        adj_curr[u_rm, v_rm] = 0
        adj_curr[v_rm, u_rm] = 0
        edges_curr.remove(edge_to_remove)
        
        # Refill greedily
        adj_curr, edges_curr = greedy_construct(adj_curr, edges_curr)
        
        # Check if improved
        if len(edges_curr) > len(edges_best):
            edges_best = edges_curr
            adj_best = adj_curr
            best_edges = len(edges_curr)
            best_graph = list(edges_curr)
            
    result = []
    for u, v in best_graph:
        result.append([u, v, 0])
        
    return result
