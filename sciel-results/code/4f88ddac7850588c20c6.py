"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Ensure positive weights to avoid log issues
    w = np.clip(weights, 1e-12, None)
    p = np.clip(prizes, 1e-12, None)
    
    # Heuristic 1: Sum-ratio (current best, score 1.0)
    total_prize = np.sum(p, axis=0)  # (100,)
    total_weight = np.sum(w, axis=0)  # (100,)
    h_sum = total_prize / (total_weight + 1e-9)  # (100,)
    
    # Heuristic 2: Geometric-mean ratio - captures items balanced across constraints
    # For each item, compute per-constraint prize/weight ratio, then geometric mean
    per_constraint_ratio = p / w  # (5, 100)
    log_mean_ratio = np.mean(np.log(per_constraint_ratio), axis=0)  # (100,)
    h_geo = np.exp(log_mean_ratio)  # (100,)
    
    # Normalize both heuristics to comparable scale
    h_sum_norm = (h_sum - h_sum.min()) / (h_sum.max() - h_sum.min() + 1e-9)
    h_geo_norm = (h_geo - h_geo.min()) / (h_geo.max() - h_geo.min() + 1e-9)
    
    # Blend: 60% sum-ratio (proven) + 40% geo-ratio (captures balance)
    h_blended = 0.6 * h_sum_norm + 0.4 * h_geo_norm
    
    # Ensure positive values
    return np.clip(h_blended, 1e-9, None)
# EVOLVE-BLOCK-END
