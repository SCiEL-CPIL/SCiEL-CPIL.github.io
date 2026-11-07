"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Combines geometric mean of per-constraint prize-to-weight ratios with
    a constraint-specific scarcity premium: identifies the most binding
    constraint for each item and boosts its score based on how scarce
    that constraint is relative to total weight.
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
    
    # Exponentiate back to get the geometric mean
    geo_mean = np.exp(log_geo_mean)
    
    # Identify the most binding constraint for each item:
    # the constraint where weight is largest fraction of total weight
    total_weight = weights.sum(axis=0)  # Shape: (100,)
    
    # Fraction of total weight in each constraint: (5, 100)
    weight_fraction = weights / (total_weight + eps)
    
    # Most binding constraint index per item: (100,)
    most_binding = weight_fraction.argmax(axis=0)
    
    # For each item, get the weight fraction of its most binding constraint
    # Use advanced indexing: weight_fraction[most_binding, arange(100)]
    item_indices = np.arange(weights.shape[1])
    most_binding_fraction = weight_fraction[most_binding, item_indices]  # Shape: (100,)
    
    # Scarcity premium: inverse of remaining capacity fraction for the most binding constraint
    # If an item uses a large fraction of its total weight in the most binding constraint,
    # that constraint is more scarce for it, so we boost its attractiveness
    # remaining_capacity_fraction = 1 - (weight_in_most_binding / capacity)
    # But we want to reward items that are efficient in their most limiting constraint
    # Use: 1 / (weight_fraction_in_most_binding + eps) as a scarcity signal
    # Items with small weight fraction in their most binding constraint get higher premium
    scarcity_premium = 1.0 / (most_binding_fraction + eps)
    
    # Normalize scarcity premium to have mean ~1 for stability
    scarcity_premium = scarcity_premium / (np.mean(scarcity_premium) + eps)
    
    # Combine: geometric mean * scarcity premium
    # The geometric mean captures overall efficiency, scarcity premium
    # boosts items that don't heavily burden their most limiting constraint
    combined = geo_mean * scarcity_premium
    
    # Ensure all values are positive
    combined = np.maximum(combined, 1e-9)
    
    return combined
# EVOLVE-BLOCK-END
