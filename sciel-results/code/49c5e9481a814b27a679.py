"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = demands.astype(float)
    pair = d[:, None] + d[None, :]
    
    # Only consider pairs that fit in the bin
    fits = pair <= capacity
    
    # Smooth Gaussian surrogate for the "near-perfect fit" objective.
    # Instead of a hard power-law (ratio**6), use exp(-sigma * (1 - ratio)^2).
    # This is smooth, differentiable, and gives a bell-shaped peak centered
    # at ratio = 1.0 (perfect fill). Smaller sigma -> wider/broader peak;
    # larger sigma -> sharper/narrower peak.
    #
    # We use sigma=8.0 which gives:
    #   ratio=1.0 -> exp(0) = 1.0
    #   ratio=0.9 -> exp(-8*0.01) = exp(-0.08) ≈ 0.923
    #   ratio=0.8 -> exp(-8*0.04) = exp(-0.32) ≈ 0.726
    #   ratio=0.6 -> exp(-8*0.16) = exp(-1.28) ≈ 0.278
    #   ratio=0.4 -> exp(-8*0.36) = exp(-2.88) ≈ 0.056
    # This strongly favors near-perfect fits while remaining smooth.
    
    ratio = np.where(fits, pair / capacity, 0.0)
    sigma = 8.0
    gaussian = np.where(fits, np.exp(-sigma * (1.0 - ratio) ** 2), 0.0)
    
    # Add a mild auxiliary signal: slightly penalize pairs where both items
    # are very small (they waste space even if they fit). This is a smooth
    # secondary term that breaks ties among similarly-ratioed pairs by
    # preferring larger individual items.
    # min_item_ratio = min(d_i, d_j) / capacity
    # We add a small bonus proportional to the geometric mean of item sizes,
    # normalized, so that larger items get a slight edge when ratios are close.
    geo_mean = np.sqrt(d[:, None] * d[None, :])
    geo_norm = geo_mean / capacity  # in [~0.13, ~0.66]
    
    # Combine: primary Gaussian peak + small geometric-mean bonus
    # The bonus is scaled down so it only acts as a tiebreaker.
    bonus_scale = 0.15
    combined = gaussian + bonus_scale * np.where(fits, geo_norm, 0.0)
    
    # Ensure positive values for fitting pairs, small value for non-fitting
    result = np.where(fits, combined, 1e-9)
    
    # Guarantee minimum positive value
    result = np.maximum(result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
