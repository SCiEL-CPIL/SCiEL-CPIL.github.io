"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial as sp
import scipy.spatial.distance as sdist
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = 300.0  # Safety margin, though actual computation is much faster
    
    n = distance_matrix.shape[0]
    
    # Reconstruct 2D coordinates from distance matrix using MDS
    # We use classical MDS (smacof is also good but classical is simpler for small n)
    # D^2 is the squared distance matrix
    D2 = distance_matrix ** 2
    
    # Compute Gram matrix
    # B = -0.5 * J * D2 * J, where J is the centering matrix
    # Alternatively, use eigendecomposition of the double-centered matrix
    H = -0.5 * np.ones((n, n))
    H[np.diag_indices(n)] = 1.0
    B = H @ D2 @ H
    
    # Eigendecomposition
    # We need the top 2 eigenvalues for 2D projection
    eigvals, eigvecs = np.linalg.eigh(B)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take top 2 dimensions
    # Scale by sqrt of eigenvalues
    V2 = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    
    points = V2
    
    # Compute Delaunay triangulation
    try:
        tri = sp.Delaunay(points)
        simplices = tri.simplices
    except Exception:
        # Fallback to inverse distance if triangulation fails
        return 1.0 / (distance_matrix + 1e-9)
    
    # Initialize heuristic matrix with a very small value
    heuristic = np.full((n, n), 1e-6)
    
    # Set high attractiveness for edges in the Delaunay triangulation
    # We use inverse squared distance to emphasize shorter edges within the triangulation
    for simplex in simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u = simplex[i]
                v = simplex[j]
                d = distance_matrix[u, v]
                if d > 0:
                    # Use a strong decay
                    val = 1.0 / (d ** 2)
                    heuristic[u, v] = val
                    heuristic[v, u] = val
    
    # Ensure symmetry and positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check time
    if time.time() - start_time > deadline * 0.9:
        # Return a safe fallback if we are running out of time (unlikely)
        return 1.0 / (distance_matrix + 1e-9)
        
    return heuristic
# EVOLVE-BLOCK-END
