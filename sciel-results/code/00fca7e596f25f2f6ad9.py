"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Use Spectral Centrality (Eigenvector Centrality) to identify peripheral vertices.
    # Peripheral vertices (low centrality) are generally safer to include in the
    # independent set as they have fewer conflicts within the dense core of the graph.
    
    # Compute the largest eigenvalue and its corresponding eigenvector.
    # np.linalg.eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
    top_eigvec = eigenvectors[:, -1]
    
    # Degrees are still a useful local feature.
    degrees = adjacency.sum(axis=1)
    
    # Score: Primarily based on negative spectral centrality (prefer peripheral).
    # Secondary: Negative degree (prefer low degree).
    # Weighting: Spectral is global, degree is local.
    scores = -top_eigvec - 0.1 * degrees
    
    return scores.astype(float)
# EVOLVE-BLOCK-END
