"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic matrix that strongly biases ants toward short edges
    while rewarding route consolidation via savings and demand-aware weighting.
    
    Key design:
    - Base: 1/d^2 (steep gradient, strong nearest-neighbor preference)
    - Savings component: reward edges that save depot trips
    - Demand weighting: prioritize consolidating high-demand customers
    - Depot edges: tuned to encourage good route starts/ends
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    
    # Base heuristic: inverse squared distance (steeper gradient than 1/d)
    # This creates strong preference for nearest neighbors, forming tight clusters
    base_heuristic = 1.0 / (dm + 1e-9) ** 2
    
    # Compute savings matrix: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    # Positive savings mean connecting i and j saves distance vs. separate trips
    depot = 0
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                s = dm[depot, i] + dm[depot, j] - dm[i, j]
                savings[i, j] = s
                savings[j, i] = s
    
    # Normalize savings to [0, 1] range
    max_savings = np.max(savings)
    if max_savings > 0:
        savings_norm = np.clip(savings, 0, max_savings) / max_savings
    else:
        savings_norm = np.zeros_like(savings)
    
    # Demand-aware weighting: edges between customers with higher combined
    # demand are more valuable for consolidation (fewer depot returns needed)
    demand_weight = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                demand_weight[i, j] = (demands[i] + demands[j]) / (2.0 * capacity)
    
    # Combine: base heuristic modulated by savings and demand structure
    # The multiplier increases attractiveness for edges that:
    # 1. Have high savings (good for consolidation)
    # 2. Connect customers with higher combined demand
    # Using a moderate boost to not overwhelm the distance-based preference
    savings_boost = 1.0 + 1.5 * savings_norm * demand_weight
    
    # Final heuristic for customer-customer edges
    heuristic = base_heuristic * savings_boost
    
    # Handle depot edges specially
    # Depot-to-customer edge attractiveness should reflect:
    # 1. Distance to depot (shorter = better)
    # 2. How well the customer can be consolidated with others
    for i in range(1, n):
        # Average savings potential of customer i with other customers
        if n > 2:
            avg_savings_i = np.mean(savings_norm[i, 1:])
        else:
            avg_savings_i = 0.0
        
        # Depot edge boost: customers with high consolidation potential
        # get slightly boosted depot edges to encourage route inclusion
        depot_boost = 1.0 + 0.8 * avg_savings_i
        
        heuristic[0, i] = base_heuristic[0, i] * depot_boost
        heuristic[i, 0] = base_heuristic[i, 0] * depot_boost
    
    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
