"""Rewritten program using spectral (Fiedler vector) approach for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def priority(adjacency):
    """
    Assign priority scores to vertices based on the Fiedler vector (second smallest
    eigenvector of the graph Laplacian). Vertices with smaller absolute Fiedler
    component are considered more peripheral/less central, making them safer to
    include in an independent set.
    
    Returns an array of scores where higher score = higher priority (selected first).
    """
    n = adjacency.shape[0]
    
    # Convert to numpy array if needed
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    # Ensure boolean/integer adjacency
    adjacency = adjacency.astype(float)
    
    # Handle edge case: empty graph
    if n == 0:
        return np.array([], dtype=float)
    
    # Compute degree vector
    degrees = adjacency.sum(axis=1)
    
    # Build Laplacian L = D - A
    # For efficiency, work with sparse representation if possible
    A_sparse = csr_matrix(adjacency)
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    # Find the Fiedler vector: eigenvector corresponding to second smallest eigenvalue
    # We want k=2 smallest eigenvalues, skip the trivial one (lambda=0)
    try:
        if n <= 2:
            # Special case for very small graphs
            return -degrees.copy()
        
        # Use eigsh to find smallest eigenvalues
        # sigma=0 shifts to find smallest eigenvalues
        # We want the 2nd smallest, so k=2, which='SM'
        eigenvalues, eigenvectors = eigsh(L, k=2, sigma=0, which='LM')
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # The Fiedler vector is the second one (index 1)
        fiedler_vec = eigenvectors[:, 1]
        
        # Use negative absolute value as priority: smaller |fiedler| -> higher priority
        # This prioritizes vertices that are less central in the graph's connectivity structure
        scores = -np.abs(fiedler_vec)
        
        # Break ties by degree (lower degree gets slightly higher priority)
        # Add a small tie-breaker based on degree
        degree_tiebreak = -degrees / (degrees.max() + 1) * 0.1
        scores = scores + degree_tiebreak
        
        return scores.astype(float)
        
    except Exception:
        # Fallback to degree-based heuristic if spectral method fails
        return -degrees.astype(float)
# EVOLVE-BLOCK-END
