import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing task.
    
    Combines a spectral clustering approach (Fiedler vector) with a direct
    waste-based heuristic. The spectral component captures global structural
    patterns in item compatibility, while the direct component provides
    local pair-level guidance.
    
    Args:
        demands: list or array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        numpy array of shape (n, n) with heuristic values.
    """
    start_time = time.time()
    DEADLINE = 320.0  # 10s safety margin before 330s kill
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
    
    # Vectorized computation of pair sums
    sums = d[:, None] + d[None, :]
    fits = sums <= capacity
    waste = capacity - sums  # >= 0 for fitting pairs
    
    # --- Direct heuristic component ---
    # H_direct[i,j] = (d_i + d_j) / (1 + waste) for fitting pairs
    # This rewards tight fits with large sums.
    safe_waste = np.where(fits, waste + 1.0, 1.0)
    H_direct = np.where(fits, sums / safe_waste, 0.0)
    
    # --- Spectral clustering component ---
    # Build a graph where edge weights are inversely proportional to residual
    # space (waste) for pairs that fit. Items with similar compatibility
    # patterns (i.e., they fit well with similar sets of other items) will
    # be close in the Fiedler vector embedding.
    
    # Edge weight: for fitting pairs, use 1/(1+waste) as weight
    # For non-fitting pairs, weight is 0
    W = np.where(fits, 1.0 / (1.0 + waste), 0.0)
    
    # Compute degree matrix
    degrees = W.sum(axis=1)
    
    # Avoid division by zero for isolated nodes
    safe_degrees = np.where(degrees > 0, degrees, 1.0)
    
    # Construct the graph Laplacian: L = D - W
    # Use the normalized Laplacian for better spectral properties
    # L_sym = I - D^(-1/2) W D^(-1/2)
    d_inv_sqrt = 1.0 / np.sqrt(safe_degrees)
    L_sym = np.eye(n) - (W * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    
    # Ensure L_sym is symmetric (numerical safety)
    L_sym = (L_sym + L_sym.T) / 2.0
    
    # Compute eigenvalues and eigenvectors
    # We want the Fiedler vector (second smallest eigenvalue's eigenvector)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        
        # Sort by eigenvalue (eigh already returns sorted)
        # The first eigenvector corresponds to the smallest eigenvalue (0 for connected graph)
        # The Fiedler vector is the second one
        if n >= 2:
            fiedler = eigenvectors[:, 1]
        else:
            fiedler = np.ones(n)
        
        # Normalize the Fiedler vector to have unit norm
        norm = np.linalg.norm(fiedler)
        if norm > 0:
            fiedler = fiedler / norm
        else:
            fiedler = np.ones(n) / np.sqrt(n)
        
        # The spectral heuristic: H_spectral[i,j] = fiedler[i] * fiedler[j]
        # This captures global compatibility structure
        H_spectral = np.outer(fiedler, fiedler)
        
        # Only keep spectral component for fitting pairs
        H_spectral = np.where(fits, H_spectral, 0.0)
        
        # Scale spectral component to be comparable to direct component
        # Normalize H_spectral to have max value of 1 for fitting pairs
        max_spectral = H_spectral[fits].max() if np.any(fits) else 1.0
        if max_spectral > 0:
            H_spectral = H_spectral / max_spectral
        else:
            H_spectral = np.zeros_like(H_spectral)
        
    except Exception:
        # Fallback if spectral computation fails
        H_spectral = np.zeros((n, n))
    
    # --- Combine the two components ---
    # Weight the direct component more heavily as it provides local guidance,
    # but the spectral component adds global structural information.
    alpha = 0.7  # Weight for direct heuristic
    beta = 0.3   # Weight for spectral heuristic
    
    # Normalize H_direct to have max value of 1 for fitting pairs
    max_direct = H_direct[fits].max() if np.any(fits) else 1.0
    if max_direct > 0:
        H_direct_norm = H_direct / max_direct
    else:
        H_direct_norm = np.zeros_like(H_direct)
    
    H_combined = alpha * H_direct_norm + beta * H_spectral
    
    # For non-fitting pairs, set to 1e-9
    H = np.where(fits, H_combined, 1e-9)
    
    # Ensure all values are positive and finite
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    # Ensure shape is correct
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        pass  # Return immediately; computation is fast anyway
    
    return H
