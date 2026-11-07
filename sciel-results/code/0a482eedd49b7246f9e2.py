import numpy as np
import time

def heuristics(demands, capacity):
    """
    Compute an item-pair attractiveness matrix for ant colony optimization
    of offline bin packing using a combination of local fit metrics and
    global spectral embedding of the item compatibility graph.
    """
    t0 = time.time()
    deadline = t0 + 320.0  # Safety margin before 330s kill

    n = len(demands)
    d = np.array(demands, dtype=float)

    # 1. Compute Pairwise Properties
    pair = d[:, None] + d[None, :]
    fits = pair <= capacity
    residual = np.where(fits, capacity - pair, 0.0)
    
    # Initialize heuristic
    h = np.full((n, n), 1e-9, dtype=float)

    # 2. Local Heuristic Components
    # Component 1: Fill ratio (higher is better)
    fill_ratio = np.where(fits, pair / capacity, 0.0)
    fill_score = fill_ratio ** 2

    # Component 2: Inverse remaining space (tighter is better)
    # Use a smooth inverse to avoid division by zero and emphasize tightness
    inv_remaining = np.where(residual > 0, 1.0 / (residual + 1.0), 0.0)
    # Scale to be comparable to fill_score (max 1.0)
    inv_score = inv_remaining * capacity * 0.01 

    # Component 3: Smooth residual bonus (peaks at 0 and min_d)
    min_d = d.min()
    sigma_tight = min_d * 0.5 + 1.0
    tight_bump = np.exp(-residual ** 2 / (2 * sigma_tight ** 2))
    
    sigma_useful = min_d * 0.5 + 1.0
    useful_bump = np.exp(-(residual - min_d) ** 2 / (2 * sigma_useful ** 2))
    
    residual_bonus = 0.5 * tight_bump + 0.2 * useful_bump

    # Combine local components
    local_h = fill_score + inv_score + residual_bonus
    local_h = np.where(fits, local_h, 0.0)

    # 3. Spectral Embedding Component
    # Build compatibility graph weights
    # Weight is high for tight fits (low residual)
    # Use exp(-residual/tau) to create a smooth decay
    tau = 10.0
    W = np.where(fits, np.exp(-residual / tau), 0.0)
    # Zero out self-loops
    np.fill_diagonal(W, 0.0)
    
    # Compute Graph Laplacian L = D - W
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    # Compute eigenvectors
    # We want the smallest non-zero eigenvalues (Fiedler vectors)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # eigh returns eigenvalues in ascending order
        # The first eigenvalue is 0 (constant vector)
        # Take the next k eigenvectors
        k = 3
        # Ensure we don't go out of bounds
        k = min(k, n - 1)
        if k > 0:
            # Eigenvectors corresponding to the smallest non-zero eigenvalues
            # eigvecs[:, 1:k+1]
            embedding = eigvecs[:, 1:k+1]
            
            # Compute pairwise Euclidean distances in the embedding space
            # dist[i, j] = ||embedding[i] - embedding[j]||^2
            # Use matrix multiplication for efficiency: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            sq_norms = np.sum(embedding ** 2, axis=1)
            dot_prod = embedding @ embedding.T
            dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * dot_prod
            # Clip small negative values due to numerical error
            dist_sq = np.maximum(dist_sq, 0.0)
            dist = np.sqrt(dist_sq)
            
            # Normalize distances to [0, 1] range for the affinity calculation
            max_dist = dist.max()
            if max_dist > 1e-6:
                dist_norm = dist / max_dist
            else:
                dist_norm = np.zeros_like(dist)
            
            # Spectral Affinity: High when items are close in spectral space
            # This captures global structural similarity
            # Use a Gaussian kernel for smooth affinity
            sigma_spec = 0.5
            spectral_affinity = np.exp(-dist_norm ** 2 / (2 * sigma_spec ** 2))
            
            # Combine local and spectral
            # Boost local heuristic by spectral affinity
            # Items that are structurally similar (close in embedding) get a boost
            # This hypothesis assumes that items in the same "community" of the
            # compatibility graph are good candidates to be packed together
            w_spec = 0.5
            h = local_h * (1.0 + w_spec * spectral_affinity)
            
        else:
            h = local_h
            
    except Exception:
        # Fallback to local heuristic if spectral computation fails
        h = local_h

    # 4. Final Cleanup
    # Ensure non-positive and non-finite values become 1e-9
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    # Don't allow placing an item after itself
    np.fill_diagonal(h, 1e-9)
    
    # Check deadline
    if time.time() > deadline - 5:
        return h

    return h
