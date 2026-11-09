"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=np.int8)
    
    # If the graph is very small or empty, handle trivially
    if n < 2:
        return np.zeros(n, dtype=np.int8)
    
    # Construct Laplacian matrix L = D - A
    # scipy.sparse.csgraph.laplacian expects adjacency matrix
    # Since input is bool/int8, we convert to float for sparse operations
    A = adjacency.astype(float)
    L = laplacian(A, normed=False)
    
    # Find the Fiedler vector (eigenvector for the 2nd smallest eigenvalue)
    # We want the smallest eigenvalues. eigs finds the largest by default.
    # We can invert the problem or use shift-invert.
    # For small n (100), dense eigendecomposition is fast and robust.
    
    # Convert to dense for simplicity and robustness for n=100
    L_dense = L.toarray() if hasattr(L, 'toarray') else L
    eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
    
    # Sort eigenvalues (eigh returns them in ascending order)
    # The 0th eigenvalue is 0 (for connected graph). The 1st is the Fiedler value.
    # If the graph is disconnected, there will be multiple zeros.
    # We take the eigenvector corresponding to the second smallest eigenvalue.
    fiedler_vec = eigenvectors[:, 1]
    
    # Initial partition based on sign of Fiedler vector
    labels = np.zeros(n, dtype=np.int8)
    labels[fiedler_vec < 0] = 1
    
    # Ensure both partitions are non-empty (edge case)
    if np.all(labels == 0) or np.all(labels == 1):
        # If all same, split in half
        labels[:n//2] = 0
        labels[n//2:] = 1
    
    # Greedy Local Search Refinement
    # Iterate through vertices and flip if it improves the cut
    # We do multiple passes until no improvement is found
    
    def get_cut_size(labels):
        return np.sum(adjacency * (labels[:, None] != labels[None, :]))
    
    current_score = get_cut_size(labels)
    improved = True
    max_iterations = 100
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        for v in range(n):
            # Calculate delta if we flip v
            # Current cut contribution from v: edges to opposite side
            # New cut contribution from v: edges to same side (which become opposite)
            # Delta = (edges to same side) - (edges to opposite side)
            
            same_mask = (labels == labels[v])
            opp_mask = (labels != labels[v])
            
            same_count = adjacency[v, same_mask].sum()
            opp_count = adjacency[v, opp_mask].sum()
            
            delta = same_count - opp_count
            
            if delta > 0:
                labels[v] = 1 - labels[v]
                current_score += delta
                improved = True
    
    return labels
# EVOLVE-BLOCK-END
