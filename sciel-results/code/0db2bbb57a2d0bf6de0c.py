"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Convert to numpy if needed
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # Compute degree matrix D
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    
    # Construct M = D + A
    # This matrix is symmetric and positive definite (if graph is connected)
    M = D + adj
    
    # Compute principal eigenvector
    # eigh returns eigenvalues in ascending order
    # We want the largest eigenvalue's eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    
    # The last column corresponds to the largest eigenvalue
    v1 = eigenvectors[:, -1]
    
    # Ensure v1 is positive (eigenvectors can have arbitrary sign)
    # Perron-Frobenius theorem states the principal eigenvector can be chosen positive
    # If the sum is negative, flip the sign
    if np.sum(v1) < 0:
        v1 = -v1
        
    # Score: Inverse of spectral centrality
    # Vertices with high spectral centrality (high v1) are "central" and likely to block many others
    # We want to pick "peripheral" vertices first, so we invert the score
    score = -v1
    
    return score
# EVOLVE-BLOCK-END
