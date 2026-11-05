"""Program using spectral centrality for Maximum Independent Set priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign scores to vertices based on eigenvector centrality.
    Vertices with lower eigenvector centrality (more peripheral) get higher priority.
    We use numpy.linalg.eig to compute the principal eigenvector of the adjacency matrix.
    
    The idea: vertices that are less central in the graph (lower eigenvector centrality)
    are less entangled with the core structure, so selecting them first leaves more
    non-neighbors available for subsequent selections, potentially yielding a larger
    independent set.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    
    # Handle edge case of empty graph
    if n == 0:
        return np.array([])
    
    # Compute eigenvalues and eigenvectors
    # For N=200, this is fast and robust
    try:
        eigenvalues, eigenvectors = np.linalg.eig(adjacency)
    except np.linalg.LinAlgError:
        # Fallback to power iteration if eig fails (rare)
        rng = np.random.RandomState(42)
        x = rng.rand(n)
        x = x / np.linalg.norm(x)
        for _ in range(200):
            x_new = adjacency @ x
            norm = np.linalg.norm(x_new)
            if norm < 1e-12:
                x_new = np.ones(n) / np.sqrt(n)
                break
            x_new = x_new / norm
            if np.linalg.norm(x_new - x) < 1e-10:
                x = x_new
                break
            x = x_new
        return -np.abs(x)

    # Find the index of the eigenvalue with the largest magnitude
    # This corresponds to the spectral radius
    max_idx = np.argmax(np.abs(eigenvalues))
    
    # Get the corresponding eigenvector
    # The eigenvector is a column in eigenvectors
    v = eigenvectors[:, max_idx]
    
    # The eigenvector might have arbitrary sign.
    # For centrality, we care about the magnitude of the influence.
    # We take the absolute value to get a non-negative centrality score.
    # This handles bipartite graphs where the principal eigenvector has mixed signs.
    centrality = np.abs(v)
    
    # Normalize to [0, 1] range to ensure consistent scale, though not strictly necessary for ranking
    c_min = centrality.min()
    c_max = centrality.max()
    if c_max - c_min > 1e-12:
        centrality = (centrality - c_min) / (c_max - c_min)
    else:
        centrality = np.zeros(n)
        
    # Lower centrality => higher priority
    # We want vertices with LOW centrality to be selected first
    # So score = -centrality
    scores = -centrality
    
    return scores
# EVOLVE-BLOCK-END
