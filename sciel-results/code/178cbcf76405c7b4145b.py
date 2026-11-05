import numpy as np
import time

def priority(adjacency):
    """
    Spectral Centrality Strategy for Maximum Independent Set.
    
    1. Compute principal eigenvector of adjacency matrix via power iteration.
    2. This identifies the "core" of the graph (high centrality) vs "periphery" (low centrality).
    3. Score = -alpha * centrality + beta * (sum of neighbor centralities) / (degree + 1)
       - Low centrality is preferred (peripheral nodes are "safer" picks, less likely to be blocked by core dynamics? 
         Actually, for MIS, we want to pick nodes that block the fewest *other* potential picks.
         High centrality nodes are hubs. Picking a hub blocks many.
         So we should AVOID high centrality nodes. Score should be low for high centrality.
         Thus: -alpha * centrality.
       
       - But we also want to pick nodes that are adjacent to hubs, so that picking them *removes* the hubs from the graph, freeing up the rest.
         This is the "hub blocking" or "opportunity cost" benefit.
         If neighbor v has high centrality, picking u removes v (if v is a hub, this is good).
         So we add a term proportional to the centrality of neighbors.
       
    4. Combine these to guide the greedy decoder.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool).astype(float)
    n = adj.shape[0]
    
    # Power iteration for principal eigenvector
    # Initialize with random or ones
    x = np.ones(n)
    x /= np.linalg.norm(x)
    
    # Convergence criteria
    tol = 1e-6
    max_iter = 100
    prev_x = None
    
    for i in range(max_iter):
        # Check time
        if time.time() - start_time > 100:
            break
            
        # Matrix-vector product
        new_x = adj @ x
        norm = np.linalg.norm(new_x)
        if norm < 1e-10:
            # Graph is empty or disconnected in a way that leads to zero?
            # If norm is 0, x stays as is (or becomes 0).
            # For MIS, if graph is empty, any score works.
            # If disconnected, power iteration on full matrix still works on the largest component?
            # Actually, if there are multiple components, the principal eigenvector is supported on the component(s) with largest lambda.
            # This is fine.
            x = new_x
            break
        
        new_x /= norm
        
        # Check convergence
        if prev_x is not None:
            diff = np.linalg.norm(new_x - prev_x)
            if diff < tol:
                x = new_x
                break
        
        prev_x = x
        x = new_x
    
    # x is now the principal eigenvector (centrality scores)
    # Normalize to [0, 1] range for easier interpretation?
    # Or just use raw values.
    # Raw values can be negative if graph is bipartite? No, principal eigenvalue is positive, eigenvector can be chosen positive.
    # Power iteration with positive init usually yields positive eigenvector for connected graphs.
    # For disconnected, some components might have 0? No, if lambda > 0, eigenvector is non-zero.
    # Let's ensure non-negative for safety in scoring logic if we use ratios.
    # Actually, eigenvector components can be negative in general, but for adjacency matrices of undirected graphs,
    # the principal eigenvector can be taken to be non-negative (Perron-Frobenius).
    # Power iteration starting from positive vector converges to the Perron vector.
    
    centrality = x
    
    # Calculate degree
    deg = adj.sum(axis=1)
    
    # Calculate neighbor centrality sum
    # sum of centralities of neighbors for each vertex
    neighbor_cent_sum = adj @ centrality
    
    # Average neighbor centrality (normalized by degree)
    # Handle degree 0
    safe_deg = deg + 1.0
    avg_neighbor_cent = neighbor_cent_sum / safe_deg
    
    # Scoring:
    # We want to pick vertices with LOW centrality (peripheral).
    # We want to pick vertices with HIGH neighbor centrality (adjacent to hubs, to block them).
    
    # Score = -w1 * centrality + w2 * avg_neighbor_cent
    
    # Scale considerations:
    # centrality is roughly in [0, 1] (normalized).
    # avg_neighbor_cent is also roughly in [0, 1].
    
    # Let's try w1 = 1.0, w2 = 1.0 initially.
    # If centrality is high, score is low (penalized).
    # If neighbor centrality is high, score is high (bonus).
    
    w1 = 1.0
    w2 = 1.0
    
    score = -w1 * centrality + w2 * avg_neighbor_cent
    
    # Safety check: ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
