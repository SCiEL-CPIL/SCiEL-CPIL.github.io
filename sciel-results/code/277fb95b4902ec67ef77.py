import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill

    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=bool)

    deg = A.sum(axis=1).astype(float)
    has_edge = deg > 0

    BIG = 1e9

    # --- Spectral approach: Fiedler vector (2nd smallest eigenvector of Laplacian) ---
    # Vertices with high magnitude in the Fiedler vector are at the periphery
    # (far from the "center" of the graph's main component), making them
    # safer to include in an independent set.
    
    A_float = A.astype(float)
    
    # Compute Laplacian L = D - A
    D = np.diag(deg)
    L = D - A_float
    
    # Compute eigendecomposition (n=200 is small, full eigh is fine)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Find the Fiedler vector: eigenvector corresponding to the 2nd smallest
        # positive eigenvalue (skip the zero eigenvalue(s) for connected components)
        # For a connected graph, the smallest eigenvalue is 0, next is the Fiedler eigenvalue
        # For disconnected graphs, we take the first positive eigenvalue
        
        positive_mask = eigvals > 1e-10
        if positive_mask.any():
            start_idx = np.where(positive_mask)[0][0]
            # Use the first positive eigenvector (Fiedler-like)
            fiedler_vec = eigvecs[:, start_idx]
        else:
            # Edge case: no positive eigenvalues (shouldn't happen for non-trivial graphs)
            fiedler_vec = np.zeros(n)
        
        # Normalize the Fiedler vector magnitude to [0, 1]
        fiedler_mag = np.abs(fiedler_vec)
        fiedler_max = fiedler_mag.max()
        if fiedler_max > 1e-12:
            fiedler_norm = fiedler_mag / fiedler_max
        else:
            fiedler_norm = np.zeros(n)
        
        # Also compute a "peripheralness" score using multiple low-frequency modes
        # to get a richer signal
        num_modes = min(5, n - start_idx) if positive_mask.any() else 0
        if num_modes > 0:
            modes = eigvecs[:, start_idx:start_idx + num_modes]
            mode_weights = np.exp(-np.arange(num_modes) * 0.5)
            mode_weights /= mode_weights.sum()
            spectral_score = np.sum((modes ** 2) * mode_weights, axis=1)
            s_max = spectral_score.max()
            if s_max > 1e-12:
                spectral_score = spectral_score / s_max
            else:
                spectral_score = np.zeros(n)
        else:
            spectral_score = np.zeros(n)
        
    except Exception:
        # Fallback if eigendecomposition fails
        fiedler_norm = np.zeros(n)
        spectral_score = np.zeros(n)
    
    # --- Fixed-point safety score: s[i] = 1 / (1 + sum_{j in nbr(i)} s[j]) ---
    # This captures "safety" — vertices whose neighbors have low total score are safer
    s = 1.0 / (deg + 1.0).copy()
    s[~has_edge] = BIG
    
    for it in range(300):
        if time.time() > deadline - 10.0:
            break
        nbr_sum = A_float @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        if np.allclose(new_s, s, rtol=1e-8, atol=1e-10):
            s = new_s
            break
        s = new_s
    
    # --- Combine spectral and safety scores ---
    # Primary signal: Fiedler vector magnitude (peripheralness)
    # Secondary signal: safety score
    # Tertiary signal: multi-mode spectral score for refinement
    
    # Normalize safety score to [0, 1] for combination
    s_noniso = s[has_edge]
    if s_noniso.size > 0:
        s_min = s_noniso.min()
        s_max_s = s_noniso.max()
        s_range = s_max_s - s_min
        if s_range > 1e-12:
            s_norm = (s - s_min) / s_range
        else:
            s_norm = np.zeros_like(s)
    else:
        s_norm = np.zeros_like(s)
    
    # Combination: spectral (Fiedler) as primary, safety as secondary
    # Weighted sum with spectral getting more weight since it directly targets
    # peripheral vertices which are safer for independent sets
    alpha = 0.6  # weight for Fiedler/peripheralness
    beta = 0.3   # weight for safety score
    gamma = 0.1  # weight for multi-mode spectral
    
    scores = alpha * fiedler_norm + beta * s_norm + gamma * spectral_score
    
    # Isolated vertices must remain at top
    scores[~has_edge] = BIG
    
    # Small tie-breaker: lower degree slightly preferred (less competition)
    if deg.max() > 0:
        deg_rank = -deg / (deg.max() + 1)
        scores = scores + 1e-6 * deg_rank
    
    if time.time() > deadline:
        return scores
    
    return scores
