"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix):
    """
    Compute heuristic matrix for ACO TSP.
    Uses inverse distance combined with a spectral gap penalty based on the Fiedler vector.
    """
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Base heuristic: inverse distance
    # Avoid division by zero
    inv_dist = 1.0 / (distance_matrix + eps)
    
    # Construct Laplacian from inverse distance weights
    # Weights W_ij = 1 / d_ij
    W = inv_dist
    
    # Row sums for diagonal of Laplacian
    row_sums = np.sum(W, axis=1)
    L = np.diag(row_sums) - W
    
    # Compute eigenvectors
    # L is symmetric, so we use eigh
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback if eigenvalue computation fails
        return np.maximum(inv_dist, 1e-9)
    
    # The Fiedler vector corresponds to the second smallest eigenvalue
    # eigenvectors are columns, sorted by eigenvalue ascending
    fiedler_vec = eigenvectors[:, 1]
    
    # Normalize Fiedler vector for stable penalty scaling
    # We want the range of fiedler_vec to be roughly [-1, 1]
    max_abs = np.max(np.abs(fiedler_vec))
    if max_abs > 1e-12:
        fiedler_vec = fiedler_vec / max_abs
    else:
        # Degenerate case, use uniform
        fiedler_vec = np.zeros(n)
        
    # Calculate difference in Fiedler coordinates for all pairs
    # f_diff[i, j] = |f_i - f_j|
    f_diff = np.abs(fiedler_vec[:, None] - fiedler_vec[None, :])
    
    # Apply spectral penalty
    # We dampen edges with large Fiedler differences
    # lambda controls the strength of the penalty
    # Since fiedler_vec is normalized to max abs 1, f_diff is in [0, 2]
    # A value of 1.5 seems reasonable to significantly dampen "long" spectral jumps
    penalty = np.exp(-1.5 * f_diff)
    
    # Combine base heuristic and penalty
    heuristic = inv_dist * penalty
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
