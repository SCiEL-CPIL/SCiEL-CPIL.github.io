import numpy as np

def heuristics(distance_matrix):
    """
    Computes a power-law inverse distance heuristic matrix for the TSP.
    
    The heuristic value for an edge (i, j) is inversely proportional to its distance
    raised to a power beta. A beta > 1 increases the bias towards shorter edges.
    """
    d = np.asarray(distance_matrix, dtype=float)
    
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
        
    n = d.shape[0]
    d_work = d.copy()
    
    # Set diagonal to infinity so 1/d becomes 0
    np.fill_diagonal(d_work, np.inf)
    
    # Avoid division by zero for off-diagonal zeros
    min_d = 1e-9
    d_work = np.where(d_work > min_d, d_work, min_d)
    
    # Exponent beta
    # Previous experiments: beta=2.0 -> 1.25, beta=3.0 -> 1.344 (worse)
    # This suggests lower beta is better. Standard is 1.0.
    beta = 1.0
    
    # Calculate heuristic
    # h = 1 / d^beta
    h = 1.0 / np.power(d_work, beta)
    
    # Diagonal is 1/inf = 0.
    
    # Normalize to mean 1 over off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    
    # Calculate mean of valid entries
    valid_h = h[mask]
    if len(valid_h) > 0:
        mean_h = np.mean(valid_h)
        if mean_h > 0:
            h = h / mean_h
        else:
            h = np.ones_like(h)
    else:
        h = np.ones_like(h)
        
    # Ensure non-finite values are handled
    h = np.where(np.isfinite(h), h, 0.0)
    
    # Entries that are not positive become 1e-9
    h = np.where(h > 0, h, 1e-9)
    
    return h
