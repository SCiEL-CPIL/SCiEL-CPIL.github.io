"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign a priority score to each vertex for greedy independent set selection.
    
    Approach:
    1. Convert adjacency to a numpy array.
    2. Use Power Iteration to find the principal eigenvector of the adjacency matrix.
       The principal eigenvector captures the "centrality" or "density" of each vertex
       in the global graph structure.
    3. Vertices with high centrality (high eigenvector values) are in dense regions.
       Selecting a vertex in a dense region removes many neighbors, which is costly.
       Therefore, we assign higher priority to vertices with LOW centrality.
    4. Score = -principal_eigenvector (normalized).
    """
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Check if graph is empty
    if not np.any(adj):
        return np.ones(n, dtype=float)
        
    # Power Iteration to find the principal eigenvector of A
    # x_{k+1} = A x_k / ||A x_k||
    
    # Initialize with degrees to accelerate convergence to the dominant mode
    degrees = adj.sum(axis=1)
    x = degrees.copy()
    
    # If all degrees are 0 (isolated vertices), handle separately
    if np.all(x == 0):
        return np.ones(n, dtype=float)
        
    # Normalize initial vector
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        x = np.random.rand(n)
        norm = np.linalg.norm(x)
    x = x / norm
    
    max_iter = 100
    for _ in range(max_iter):
        x_new = adj @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new = x_new / norm
        
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-8:
            x = x_new
            break
        x = x_new
        
    # The principal eigenvector x has values roughly proportional to "centrality".
    # We want to pick vertices with LOW centrality first.
    # So we invert the score.
    
    # Normalize the score to [0, 1] for consistency, then invert.
    # High score = pick first.
    
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-12:
        # All vertices have similar centrality (e.g., regular graph)
        # Fall back to random or degree-based tie-breaking
        # In a regular graph, all degrees are same. Any order is fine.
        # We return a uniform score (which results in arbitrary order, effectively random)
        # Or we can use the index to break ties deterministically but arbitrarily.
        # Returning 1.0 for all is fine, the decoder will handle it.
        return np.ones(n, dtype=float)
        
    # Normalize x to [0, 1]
    x_norm = (x - x_min) / (x_max - x_min)
    
    # Invert: low centrality -> high score
    score = 1.0 - x_norm
    
    return score
# EVOLVE-BLOCK-END
