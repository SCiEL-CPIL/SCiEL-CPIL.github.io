"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    n = 20
    best_edges_count = 0
    best_adj = None
    
    # Precompute indices for masking
    indices = np.arange(n)
    # M[u, v, w] is True if w != u and w != v
    # u is axis 0, v is axis 1, w is axis 2
    # w != u: indices[None, None, :] != indices[:, None, None]
    # w != v: indices[None, None, :] != indices[None, :, None]
    mask_w_not_u = (indices[None, None, :] != indices[:, None, None])
    mask_w_not_v = (indices[None, None, :] != indices[None, :, None])
    mask_valid_w = mask_w_not_u & mask_w_not_v
    
    # Try multiple random restarts to escape local optima
    for seed in range(50):
        random.seed(seed)
        np.random.seed(seed)
        adj = np.zeros((n, n), dtype=np.int8)
        
        # Greedy loop
        while True:
            # Compute common neighbor counts D[u, w] = |N(u) \cap N(w)|
            D = adj @ adj
            
            # Construct T1[u, v, w] = D[u, w] + adj[w, v]
            # D[:, None, :] has shape (n, 1, n) -> indices u, _, w
            # adj.T[None, :, :] has shape (1, n, n) -> indices _, v, w (since adj.T[v, w] = adj[w, v])
            T1 = D[:, None, :] + adj.T[None, :, :]
            
            # Construct T2[u, v, w] = D[v, w] + adj[u, w]
            # adj[:, None, :] has shape (n, 1, n) -> indices u, _, w
            # D.T[None, :, :] has shape (1, n, n) -> indices _, v, w (since D is symmetric, D.T[v, w] = D[v, w])
            T2 = adj[:, None, :] + D.T[None, :, :]
            
            # Apply mask: if w == u or w == v, set value to -1 (so max ignores it)
            # We want max over w. If masked out, it shouldn't contribute to max > 2.
            # So set masked values to a very small number or 0?
            # The condition is Max <= 2.
            # If we set masked values to 0, and all valid w have values <= 2, max is <= 2.
            # If we set masked values to -1, same.
            # However, if there are no valid w (e.g. n=2), max would be -1 or 0.
            # For n=20, there are always valid w.
            
            T1_masked = np.where(mask_valid_w, T1, -1)
            T2_masked = np.where(mask_valid_w, T2, -1)
            
            Max1 = T1_masked.max(axis=2)
            Max2 = T2_masked.max(axis=2)
            
            # Valid edges: adj[u, v] == 0, Max1[u, v] <= 2, Max2[u, v] <= 2
            # Also u != v
            valid_mask = (adj == 0) & (Max1 <= 2) & (Max2 <= 2)
            # Exclude diagonal
            np.fill_diagonal(valid_mask, False)
            
            # Find indices of valid edges
            valid_indices = np.argwhere(valid_mask)
            
            if len(valid_indices) == 0:
                break
            
            # Pick a random valid edge
            idx = random.randint(0, len(valid_indices) - 1)
            u, v = valid_indices[idx]
            
            # Add edge
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
