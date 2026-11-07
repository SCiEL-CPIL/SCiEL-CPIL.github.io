"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    d = np.array(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Set diagonal to large value
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    
    # Sparsify: Keep only top K nearest neighbors for each node
    K = 10
    eta = np.full_like(d_safe, 1e-9)
    
    for i in range(n):
        # Get indices of K smallest distances
        # argsort returns indices that would sort the row
        indices = np.argsort(d_safe[i])[:K]
        # Set heuristic for these neighbors
        # Use alpha=2 for the values
        for j in indices:
            eta[i, j] = 1.0 / (d_safe[i, j] ** 2)
            
    # Ensure finite and positive
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
