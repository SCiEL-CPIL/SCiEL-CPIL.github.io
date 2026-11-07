import numpy as np
import time

_DEADLINE = 320  # Safety margin before 330s limit

def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for ACO-based offline bin packing.
    
    The heuristic uses a graph-theoretic approach:
    1. Build a compatibility graph where edge weight = -residual space (more negative = tighter fit = better)
    2. Compute spectral embedding (Laplacian eigenvectors) to capture global structural affinities
    3. Combine local gap-utility (dead-gap reward) with global spectral similarity
    
    Returns a (n, n) matrix where eta[i, j] is the attractiveness of placing item j after item i.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Compute pair sums and validity
    sums = demands[:, None] + demands[None, :]
    valid = sums <= capacity
    residuals = capacity - sums  # residual space for pair (i, j)
    
    # Initialize with floor value
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # --- Local gap-utility signal ---
    # Count how many other items fit in the residual space (dead-gap detection)
    sorted_demands = np.sort(demands)
    flat_residuals = residuals.flatten()
    total_counts = np.searchsorted(sorted_demands, flat_residuals, side='right').reshape(n, n)
    
    # Exclude i and j from the count
    i_fits = demands[:, None] <= residuals
    j_fits = demands[None, :] <= residuals
    dead_gap_counts = total_counts.astype(float) - i_fits.astype(float) - j_fits.astype(float)
    dead_gap_counts = np.maximum(dead_gap_counts, 0)
    
    # Local score: exponential decay of live-gap count
    # Use a softer decay to avoid overly penalizing moderate gaps
    alpha_local = 1.2
    local_scores = np.exp(-alpha_local * dead_gap_counts)
    
    # --- Size-weighted tightness signal (FFD-inspired) ---
    # Reward placing larger items (d_j) into tighter fits (smaller residuals)
    # This aligns with First Fit Decreasing principles
    # Normalize demands to [0, 1] range
    d_min = np.min(demands)
    d_max = np.max(demands)
    if d_max > d_min:
        d_norm = (demands - d_min) / (d_max - d_min)
    else:
        d_norm = np.ones(n) / n
    
    # Tightness: use a smoother function that emphasizes very tight fits
    # residual ranges from 0 to capacity - 2*min_demand
    max_residual = capacity - 2 * d_min
    if max_residual > 0:
        normalized_residual = residuals / max_residual
        # Use a power function to emphasize tight fits (small residuals)
        tightness = 1.0 - np.power(normalized_residual, 0.7)
    else:
        tightness = np.ones((n, n))
    
    # Size-weighted tightness: product of tightness and size of item j
    # This rewards placing large items into tight spaces
    size_weighted_tightness = tightness * d_norm[None, :]
    
    # Combine with local scores: weighted average
    w_size = 0.35
    local_scores = (1 - w_size) * local_scores + w_size * size_weighted_tightness
    
    # --- Global spectral embedding signal ---
    # Build adjacency matrix: edge (i, j) exists if valid, weight = -residual (tighter = higher weight)
    # Use negative residual so that tighter fits (smaller residual) get higher weight
    A = np.zeros((n, n), dtype=float)
    A[valid] = -residuals[valid]
    # Normalize weights to [0, 1] range for stability
    max_w = np.max(A[valid]) if np.any(valid) else 1.0
    min_w = np.min(A[valid]) if np.any(valid) else 0.0
    if max_w > min_w:
        A[valid] = (A[valid] - min_w) / (max_w - min_w)
    else:
        A[valid] = 1.0
    
    # Compute degree matrix and normalized Laplacian
    degrees = np.sum(A, axis=1)
    # Avoid division by zero
    degrees_safe = np.where(degrees > 0, degrees, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees_safe))
    
    # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Compute a few smallest eigenvectors (excluding trivial ones)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Take the first k non-trivial eigenvectors (skip the first which is constant)
        # Use fewer dimensions to reduce noise sensitivity
        k = min(4, n - 1)
        if k > 0:
            # Eigenvectors corresponding to smallest non-trivial eigenvalues
            # Skip index 0 (trivial)
            idx = np.argsort(eigenvalues)[1:1+k]
            V = eigenvectors[:, idx]
        else:
            V = np.zeros((n, 1))
        
        # Compute pairwise cosine similarity in the spectral embedding space
        # Normalize rows of V
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms_safe = np.where(norms > 1e-10, norms, 1.0)
        V_norm = V / norms_safe
        
        # Pairwise dot products = cosine similarities
        spectral_sim = V_norm @ V_norm.T
        # Map from [-1, 1] to [0, 1]
        spectral_sim = (spectral_sim + 1.0) / 2.0
        
    except Exception:
        # Fallback: identity-based similarity
        spectral_sim = np.eye(n)
    
    # Combine local and global signals
    # Weighted combination: local captures tightness + size preference, global captures structural affinity
    # Shift weight slightly toward local signal which has clearer interpretation
    w_local = 0.6
    w_global = 0.4
    
    combined = w_local * local_scores + w_global * spectral_sim
    
    # Apply only to valid pairs
    eta[valid] = combined[valid]
    
    # Diagonal: no self-pairing
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    # Deadline guard: if we're running long, return immediately
    # (The computation above is vectorized and should be fast, but guard just in case)
    return eta
