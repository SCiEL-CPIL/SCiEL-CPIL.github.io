import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing task.
    
    Combines the squared-inverse-waste direct heuristic with a spectral
    (Fiedler vector) component built from a multi-scale compatibility graph
    with greedy-matching boost.
    
    Args:
        demands: list or array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        numpy array of shape (n, n) with heuristic values.
    """
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
    
    # Vectorized computation of pair sums
    sums = d[:, None] + d[None, :]
    fits = sums <= capacity
    waste = capacity - sums  # >= 0 for fitting pairs
    
    # --- Direct heuristic component (squared inverse waste) ---
    eps = 1e-6
    safe_waste = np.where(fits, waste + eps, 1.0)
    H_direct = np.where(fits, 1.0 / (safe_waste ** 2), 0.0)
    
    # --- Multi-scale spectral graph ---
    # Use a smooth exponential kernel for continuous compatibility weights
    # This avoids hard thresholds and provides a smooth surrogate
    sigma = 15.0  # Kernel bandwidth controls the scale of compatibility
    W = np.where(fits, np.exp(-waste / sigma), 0.0)
    
    # --- Greedy matching boost ---
    # Find high-quality pairs and boost their edge weights
    # This guides the spectral structure toward proven good pairings
    if n >= 2:
        # Sort all fitting pairs by waste (ascending = best first)
        fit_indices = np.where(fits)
        if len(fit_indices[0]) > 0:
            pair_waste = waste[fit_indices]
            # Get indices of best pairs
            sorted_idx = np.argsort(pair_waste)
            used = np.zeros(n, dtype=bool)
            boost = 0.0
            for idx in sorted_idx:
                i = fit_indices[0][idx]
                j = fit_indices[1][idx]
                if not used[i] and not used[j]:
                    w = waste[i, j]
                    # Boost proportional to inverse waste, capped
                    boost_val = min(5.0 / (1.0 + w), 2.0)
                    W[i, j] += boost_val
                    W[j, i] += boost_val
                    used[i] = True
                    used[j] = True
    
    # Ensure symmetry
    W = (W + W.T) / 2.0
    
    # Compute degree matrix
    degrees = W.sum(axis=1)
    safe_degrees = np.where(degrees > 0, degrees, 1.0)
    
    # Construct symmetric normalized Laplacian
    d_inv_sqrt = 1.0 / np.sqrt(safe_degrees)
    L_sym = np.eye(n) - (W * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    
    # Ensure symmetry (numerical safety)
    L_sym = (L_sym + L_sym.T) / 2.0
    
    # Compute Fiedler vector
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        
        if n >= 2:
            fiedler = eigenvectors[:, 1]
        else:
            fiedler = np.ones(n)
        
        # Use absolute value since sign is arbitrary
        fiedler = np.abs(fiedler)
        
        # Normalize
        norm = np.linalg.norm(fiedler)
        if norm > 0:
            fiedler = fiedler / norm
        else:
            fiedler = np.ones(n) / np.sqrt(n)
        
        # Spectral heuristic: outer product
        H_spectral = np.outer(fiedler, fiedler)
        
        # Only keep for fitting pairs
        H_spectral = np.where(fits, H_spectral, 0.0)
        
        # Scale to be comparable to direct component
        max_spectral = H_spectral[fits].max() if np.any(fits) else 1.0
        if max_spectral > 0:
            H_spectral = H_spectral / max_spectral
        else:
            H_spectral = np.zeros_like(H_spectral)
            
    except Exception:
        H_spectral = np.zeros((n, n))
    
    # --- Combine components ---
    max_direct = H_direct[fits].max() if np.any(fits) else 1.0
    if max_direct > 0:
        H_direct_norm = H_direct / max_direct
    else:
        H_direct_norm = np.zeros_like(H_direct)
    
    # Tune weights: give more weight to spectral to leverage global structure
    alpha = 0.55  # Direct heuristic weight
    beta = 0.45   # Spectral heuristic weight
    
    H_combined = alpha * H_direct_norm + beta * H_spectral
    
    # For non-fitting pairs, set to 1e-9
    H = np.where(fits, H_combined, 1e-9)
    
    # Ensure all values are positive and finite
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
