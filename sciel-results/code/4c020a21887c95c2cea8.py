"""Heuristic for ACO bin packing: tight-fit complementarity with structural centrality."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Compatibility matrix: fits[i,j] = True if demands[i] + demands[j] <= capacity
    fits = (demands[:, None] + demands[None, :]) <= capacity
    
    # Waste after placing both items: capacity - demands[i] - demands[j]
    waste = capacity - demands[:, None] - demands[None, :]
    
    # Degree of each item: number of items it can pair with
    degree = fits.sum(axis=1).astype(float)
    # Avoid division by zero
    degree_safe = np.maximum(degree, 1.0)
    
    # Inverse degree: constrained items get higher weight
    # An item j that is constrained (few partners) should be more attractive to place
    inv_degree_j = 1.0 / degree_safe[None, :]  # shape (1, n)
    
    # Sharp tightness: 1/waste strongly rewards near-exact fills
    eps = 1e-6
    tightness = 1.0 / (waste + eps)  # shape (n, n)
    
    # Size similarity: items of similar size tend to tile better in remaining space
    # cos-similarity of sizes as a soft bonus
    d_i = demands[:, None]
    d_j = demands[None, :]
    sim = (d_i * d_j) / (np.maximum(d_i * d_j, 1e-6) + 1.0)  # in (0,1), ~1 when similar
    
    # Complementarity bonus: prefer pairs whose combined size is close to capacity
    # This is already captured by tightness, but we add a secondary signal:
    # the "fill ratio" of the pair relative to capacity
    fill_ratio = (d_i + d_j) / capacity  # in (0, 2), but for fitting pairs in (0, 1]
    # Reward fill ratios close to 1.0 (bin nearly full)
    fill_bonus = 1.0 / (1.0 + np.abs(fill_ratio - 1.0) + eps)
    
    # Combine: primary signal is sharp tightness, modulated by structural centrality
    # Secondary: size similarity and fill ratio as soft complements
    heuristic = tightness * inv_degree_j * (0.5 + 0.5 * sim) * fill_bonus
    
    # For non-fitting pairs, set to very small value
    heuristic = np.where(fits, heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
