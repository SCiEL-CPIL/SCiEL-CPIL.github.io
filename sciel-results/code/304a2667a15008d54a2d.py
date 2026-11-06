"""Spectral partitioning with local search refinement for Maximum Cut."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array if needed
    A = np.asarray(adjacency, dtype=np.float64)
    
    # Compute degree vector
    degrees = A.sum(axis=1)
    
    # Build graph Laplacian: L = D - A
    D = np.diag(degrees)
    L = D - A
    
    # Compute eigenvalues and eigenvectors
    # For n=100, this is fast enough
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to simple alternating pattern
        return np.arange(n) % 2
    
    # The Fiedler vector is the eigenvector corresponding to the second-smallest eigenvalue
    # eigh returns eigenvalues in ascending order, so index 1 is the second smallest
    fiedler_vector = eigenvectors[:, 1]
    
    # Create initial partition: vertices with positive Fiedler values go to 0, negative to 1
    # We can use the median or zero as threshold; median ensures balanced partition
    # For max cut, balance isn't strictly required, but using zero threshold is standard
    labels = np.where(fiedler_vector >= 0, 0, 1).astype(int)
    
    # Refine with 1-flip local search
    # Multiple passes to ensure convergence
    max_passes = 10
    for _ in range(max_passes):
        improved = False
        for v in range(n):
            # Count edges going to same side vs other side
            same_side = (labels == labels[v])
            other_side = (labels != labels[v])
            
            # Number of edges from v to same side
            edges_same = A[v][same_side].sum()
            # Number of edges from v to other side
            edges_other = A[v][other_side].sum()
            
            # If flipping v increases the cut, flip it
            if edges_same > edges_other:
                labels[v] = 1 - labels[v]
                improved = True
        
        if not improved:
            break
    
    return labels
# EVOLVE-BLOCK-END
