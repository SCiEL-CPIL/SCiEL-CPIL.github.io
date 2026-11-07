"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic matrix using a relaxed min-cost flow / LP approach
    to approximate optimal routing structure, then blend with distance-based
    heuristic.
    
    The idea: solve a transportation-like LP where we want to assign customers
    to routes (depot returns) minimizing total distance, with capacity constraints.
    We approximate this by considering which edges are 'natural' in a low-cost
    routing.
    
    For practical purposes with 50 nodes, we use a savings-based approach combined
    with a smoothed inverse-distance that encodes route consolidation structure.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Base heuristic: inverse distance (closer = more attractive)
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Savings-based component: reward edges that save distance vs. separate trips
    # Savings(i, j) = d(0,i) + d(0,j) - d(i,j)
    depot = 0
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                s = distance_matrix[depot, i] + distance_matrix[depot, j] - distance_matrix[i, j]
                savings[i, j] = s
                savings[j, i] = s
    
    # Normalize savings to be positive and comparable
    max_savings = np.max(savings)
    if max_savings > 0:
        savings_norm = np.clip(savings, 0, max_savings) / max_savings
    else:
        savings_norm = np.zeros_like(savings)
    
    # Demand-aware weighting: edges between high-demand customers are more valuable
    # for route consolidation (fewer depot returns)
    demand_weight = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and i > 0 and j > 0:
                # Higher combined demand means more savings from consolidating
                demand_weight[i, j] = (demands[i] + demands[j]) / (2 * capacity)
    
    # Combine: base distance heuristic modulated by savings and demand structure
    # The heuristic should be higher for edges that are:
    # 1. Short (low distance)
    # 2. Have high savings (good for route consolidation)
    # 3. Connect customers with higher combined demand (more valuable to consolidate)
    
    # Scale savings contribution
    savings_component = 1.0 + 2.0 * savings_norm * demand_weight
    
    # Final heuristic: blend of inverse distance and savings-enhanced structure
    heuristic = base_heuristic * savings_component
    
    # Ensure depot edges are handled properly
    # Depot-to-customer edges should also reflect savings structure
    for i in range(1, n):
        # The attractiveness of going from depot to i depends on how well i fits
        # into a route. We use the average savings potential of i with others.
        if i > 0:
            avg_savings_i = np.mean(savings[i, 1:]) if n > 2 else 0
            savings_factor_i = 1.0 + 0.5 * (avg_savings_i / max_savings if max_savings > 0 else 0)
            heuristic[0, i] = base_heuristic[0, i] * savings_factor_i
            heuristic[i, 0] = base_heuristic[i, 0] * savings_factor_i
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
