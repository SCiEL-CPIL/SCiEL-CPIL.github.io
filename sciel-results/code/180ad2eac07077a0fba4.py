"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the ant colony heuristic.
    
    Strategy: Geometric alignment with efficient frontier direction blended with
    harmonic-mean ratio.
    
    The geometric alignment score measures how well each item's prize-to-weight
    profile aligns with the global efficient frontier direction. This prioritizes
    items that are proportionally efficient across all constraints relative to
    the average resource efficiency.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    # Handle input shapes: prizes and weights should be (5, 100)
    if prizes.ndim == 2:
        p = prizes  # (5, 100)
    else:
        # If prizes is 1D (100,), broadcast to (5, 100)
        p = np.tile(prizes, (weights.shape[0], 1))
    
    w = weights  # (5, 100)
    eps = 1e-9
    n_constraints = p.shape[0]  # 5
    n_items = p.shape[1]  # 100
    
    # --- Component 1: Harmonic mean of per-constraint ratios ---
    ratios = p / (w + eps)  # (5, 100)
    inv_ratios = 1.0 / (ratios + eps)  # (5, 100)
    mean_inv = np.mean(inv_ratios, axis=0)  # (100,)
    h_harmonic = 1.0 / (mean_inv + eps)  # (100,)
    
    # --- Component 2: Geometric alignment score ---
    # The efficient frontier direction is the vector of mean prizes per unit weight
    # across all constraints. Items whose prize/weight profile aligns well with this
    # direction are more attractive.
    
    # Mean prize per unit weight per constraint: (5,)
    mean_ratio_per_constraint = np.mean(ratios, axis=1)  # (5,)
    
    # Normalize to get the efficient frontier direction
    norm = np.linalg.norm(mean_ratio_per_constraint)
    if norm > eps:
        frontier_dir = mean_ratio_per_constraint / norm  # (5,)
    else:
        frontier_dir = np.ones(n_constraints) / np.sqrt(n_constraints)
    
    # For each item, compute its prize/weight vector and its alignment with the frontier
    # Item i's vector: ratios[:, i] (5,)
    # Cosine similarity with frontier_dir: dot(ratios[:, i], frontier_dir) / (||ratios[:, i]|| * ||frontier_dir||)
    
    # Compute norms of each item's ratio vector: (100,)
    item_norms = np.linalg.norm(ratios, axis=0)  # (100,)
    
    # Dot product of each item's ratio vector with frontier_dir: (100,)
    # dot(ratios[:, i], frontier_dir) = sum_c ratios[c, i] * frontier_dir[c]
    dot_products = ratios.T @ frontier_dir  # (100,)
    
    # Cosine similarity: (100,)
    cos_sim = dot_products / (item_norms * norm + eps)  # (100,)
    
    # Clip to [0, 1] since we want positive scores
    cos_sim = np.clip(cos_sim, 0, 1)
    
    # --- Blend the two components ---
    # Normalize both to [0, 1] range
    def normalize_positive(arr):
        arr = np.where(arr > 0, arr, eps)
        mn = np.min(arr)
        mx = np.max(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.ones_like(arr)
    
    h1 = normalize_positive(h_harmonic)  # (100,)
    h2 = normalize_positive(cos_sim)  # (100,)
    
    # Blend: 50/50
    alpha = 0.5
    h = alpha * h1 + (1 - alpha) * h2
    
    # Ensure positive values as per contract
    h = np.where(h > 0, h, 1e-9)
    
    # Scale to reasonable range
    max_h = np.max(h)
    if max_h > 0:
        h = h / max_h
    
    # Ensure minimum value
    h = np.maximum(h, 1e-9)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return a simple fallback
        return np.ones(n_items) * 1e-9
    
    return h
# EVOLVE-BLOCK-END
