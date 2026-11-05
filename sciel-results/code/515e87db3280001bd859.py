"""Program using spectral centrality for Maximum Independent Set priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign scores to vertices based on the principal eigenvector of the Laplacian matrix.
    Vertices with higher values in the principal Laplacian eigenvector are considered more central (hubs).
    We prioritize vertices with LOWER values (more peripheral) to maximize the independent set size.
    
    The Laplacian matrix L = D - A captures both degree and connectivity structure.
    The principal eigenvector of L highlights nodes that are highly connected to other highly connected nodes (hubs).
    By negating the absolute value of this eigenvector, we give higher priority to peripheral nodes.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Compute Degree matrix
    degrees = np.sum(adjacency, axis=1)
    
    # Compute Laplacian L = D - A
    # D is diagonal matrix with degrees on diagonal
    # L = diag(degrees) - adjacency
    # To avoid creating a huge sparse matrix explicitly if n is large, we can do matrix-vector products directly.
    # But n=200 is small, so we can just build L.
    L = np.diag(degrees) - adjacency
    
    # Power iteration to find principal eigenvector of L
    rng = np.random.RandomState(42)
    x = rng.rand(n)
    x = x / np.linalg.norm(x)
    
    for _ in range(200):
        x_new = L @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            # Graph is a set of isolated nodes (L=0) or similar
            x_new = np.ones(n) / np.sqrt(n)
            break
        x_new = x_new / norm
        if np.linalg.norm(x_new - x) < 1e-10:
            x = x_new
            break
        x = x_new
        
    # Ensure non-negative for consistent scoring (eigenvector sign is arbitrary)
    x = np.abs(x)
    
    # Normalize to [0, 1]
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min > 1e-12:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros(n)
        
    # Lower value in principal Laplacian eigenvector => higher priority
    # So score = -x
    scores = -x
    
    return scores
# EVOLVE-BLOCK-END
