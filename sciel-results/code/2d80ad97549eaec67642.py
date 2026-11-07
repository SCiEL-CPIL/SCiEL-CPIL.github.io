"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    W = np.asarray(demands, dtype=np.float64)
    n = len(W)
    
    # Compute all pair sums
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= capacity
    
    # Gap: how much space is left in the bin after placing i and j
    # Smaller gap = better fit
    gap = capacity - pair_sums
    
    # For fitting pairs: reward tight packing
    # Use 1/(gap + small_eps) to strongly prefer tight fits
    # But also add a term that prefers larger items (harder to place)
    # h = alpha * (1/(gap+eps)) + beta * (size_i + size_j) / capacity
    
    # Normalize gap to [0, 1] range for stability
    # gap ranges from 0 to capacity-40 (min pair sum is 40)
    eps = 1e-9
    inv_gap = np.where(fit_mask, 1.0 / (gap + eps), 0.0)
    
    # Prefer larger items (they're more constraining)
    size_score = (W[:, None] + W[None, :]) / capacity
    
    # Combine: tight fit is most important, but larger items get a bonus
    # Use a weighted combination
    alpha = 1.0
    beta = 0.3
    
    H = alpha * inv_gap + beta * size_score
    
    # Set non-fitting pairs to very small value
    H = np.where(fit_mask, H, 1e-9)
    
    # Ensure no negative or zero values for fitting pairs
    H = np.maximum(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
