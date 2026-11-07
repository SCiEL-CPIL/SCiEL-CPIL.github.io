import numpy as np


def heuristics(distance_matrix):
    d = np.array(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Set diagonal to large value to avoid self-loops
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    
    # Sparsify: keep only top K nearest neighbors for each node
    K = 12
    eta = np.full_like(d_safe, 1e-9)
    
    for i in range(n):
        indices = np.argsort(d_safe[i])[:K]
        for j in indices:
            eta[i, j] = 1.0 / (d_safe[i, j] ** 3)
    
    # Symmetrize: take max of eta[i,j] and eta[j,i]
    eta = np.maximum(eta, eta.T)
    
    # Ensure finite and positive
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
