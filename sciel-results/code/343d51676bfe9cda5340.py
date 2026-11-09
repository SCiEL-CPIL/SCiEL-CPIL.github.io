"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    n = 20
    best_edges_count = 0
    best_adj = None
    
    # Try multiple random orderings to escape local optima
    for seed in range(20):
        random.seed(seed)
        adj = np.zeros((n, n), dtype=np.int8)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if adj[u, v]:
                continue
            
            # Vectorized check for K2,3 constraint
            # cuw[w] = |N(u) ∩ N(w)|
            # cvw[w] = |N(v) ∩ N(w)|
            cuw = adj[u, :] @ adj
            cvw = adj[v, :] @ adj
            
            # Mask out u and v
            mask = np.ones(n, dtype=bool)
            mask[u] = False
            mask[v] = False
            
            # Check if adding (u, v) violates constraints for any w
            # For pair (u, w): new common neighbors = cuw[w] + (1 if v in N(w) else 0)
            # adj[mask, v] is a vector where element w is 1 if v in N(w)
            if np.any(cuw[mask] + adj[mask, v] > 2):
                continue
                
            # For pair (v, w): new common neighbors = cvw[w] + (1 if u in N(w) else 0)
            if np.any(cvw[mask] + adj[mask, u] > 2):
                continue
            
            # If valid, add the edge
            adj[u, v] = 1
            adj[v, u] = 1
        
        # Count edges
        edge_count = np.sum(adj) // 2
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_adj = adj.copy()
            
    # Convert best adjacency matrix to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
