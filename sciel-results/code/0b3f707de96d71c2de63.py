"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Combines geometric mean of per-constraint prize-to-weight ratios with a
    constraint-specific scarcity premium: for each item, identify the most
    binding constraint (where its weight is the largest fraction of total weight)
    and boost its score based on the inverse remaining capacity fraction for that
    specific constraint.
    """
    eps = 1e-9
    capacity = 25.0
    
    # Per-constraint prize/weight ratio for each item
    # Shape: (5, 100)
    ratios = prizes / (weights + eps)
    
    # Geometric mean across constraints: (prod(ratios_c))^(1/5)
    # Using log-space for numerical stability
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    geo_mean = np.exp(log_geo_mean)
    
    # Identify the most binding constraint for each item
    # The constraint where the item's weight is the largest fraction of total weight
    total_weight_per_item = weights.sum(axis=0)  # Shape: (100,)
    
    # Fraction of each constraint's weight in the item's total weight
    weight_fractions = weights / (total_weight_per_item + eps)  # Shape: (5, 100)
    
    # Most binding constraint index for each item
    most_binding_idx = weight_fractions.argmax(axis=0)  # Shape: (100,)
    
    # Get the weight fraction of the most binding constraint for each item
    most_binding_fraction = weight_fractions[most_binding_idx, np.arange(len(most_binding_idx))]
    
    # Remaining capacity fraction for the most binding constraint
    # If an item uses fraction f of its total weight in constraint c, and total capacity is 25,
    # the remaining capacity fraction is (1 - f * total_weight / capacity)
    # But more directly: the item consumes weights[c,i] from capacity 25
    # So remaining fraction = 1 - weights[most_binding_idx, i] / capacity
    most_binding_weights = weights[most_binding_idx, np.arange(len(most_binding_idx))]
    remaining_fraction = 1.0 - most_binding_weights / capacity
    
    # Clip to avoid division by zero or negative values
    remaining_fraction = np.clip(remaining_fraction, eps, 1.0)
    
    # Scarcity premium: inverse of remaining capacity fraction
    # Items that consume more of their most binding constraint get a higher premium
    scarcity_premium = 1.0 / remaining_fraction
    
    # Normalize scarcity premium to have mean ~1 for stable blending
    scarcity_premium = scarcity_premium / (scarcity_premium.mean() + eps)
    
    # Blend geometric mean with scarcity premium
    # Use a weighted combination: 0.7 * geo_mean + 0.3 * geo_mean * scarcity_premium
    alpha = 0.3
    blended = geo_mean * (1.0 + alpha * (scarcity_premium - 1.0))
    
    # Ensure all values are positive
    blended = np.maximum(blended, 1e-9)
    
    return blended
# EVOLVE-BLOCK-END
