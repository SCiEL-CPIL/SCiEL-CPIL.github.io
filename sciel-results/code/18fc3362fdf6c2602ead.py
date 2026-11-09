"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO.
    Uses a Lagrangian Relaxation-based approach: identify edges that are
    locally optimal under relaxed capacity constraints, then map these to
    high attractiveness values.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    demands = np.asarray(demands, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    
    n = distance_matrix.shape[0]
    eps = 1e-9
    safe_dist = distance_matrix + eps
    
    # Base heuristic: inverse distance
    base_heuristic = 1.0 / safe_dist ** 2
    
    # Lagrangian Relaxation approach:
    # 1. Solve a relaxed problem that ignores capacity but penalizes load
    # 2. Use dual variables to identify edges that are "capacity-friendly"
    
    # Estimate capacity utilization patterns
    # For each customer, compute how "expensive" it is to include in a route
    # based on its demand relative to typical route load
    
    # Compute demand-weighted distance: edges to high-demand nodes are more
    # valuable if they can be packed efficiently
    avg_demand = np.mean(demands[1:]) if n > 1 else 1.0
    if avg_demand > 0:
        demand_ratio = demands / avg_demand
    else:
        demand_ratio = np.ones(n)
    demand_ratio[0] = 1.0  # Depot
    
    # Lagrangian penalty: penalize edges that would lead to capacity violations
    # An edge (i, j) is "penalized" if visiting j after i would likely exceed
    # capacity. We approximate this by considering the demand of j relative
    # to remaining capacity in a typical route.
    
    # For each node, compute a "capacity pressure" score:
    # High demand nodes create more pressure, so edges to/from them should
    # be weighted based on how well they fit in typical routes.
    
    # Compute a soft capacity constraint satisfaction score for each edge
    # Using a smooth surrogate: instead of hard capacity check, use a
    # sigmoid-like function that smoothly transitions between "fits" and "doesn't fit"
    
    # Typical route has about capacity/avg_demand customers
    typical_route_size = capacity / avg_demand if avg_demand > 0 else n
    
    # For each pair (i, j), estimate if j fits after i in a typical route
    # This is approximate: we consider the demand of j and the "average"
    # remaining capacity when transitioning from i to j
    
    # Create a demand compatibility matrix
    # demand_compat[i, j] = 1 if demand[j] fits well after demand[i]
    # Use a smooth approximation: sigmoid((capacity - demand[i] - demand[j]) / threshold)
    
    threshold = capacity * 0.3  # Smoothness parameter
    demand_sum = demands[:, np.newaxis] + demands[np.newaxis, :]
    # For depot (node 0), demand is 0, so this just checks if demand[j] <= capacity
    # For customer i, we check if demand[i] + demand[j] <= capacity
    
    # Smooth indicator: 1 if fits, 0 if doesn't, with smooth transition
    margin = capacity - demand_sum
    # Use a smooth step function: 0.5 * (1 + tanh(margin / threshold))
    demand_compat = 0.5 * (1.0 + np.tanh(margin / threshold))
    
    # The depot row and column should always be 1 (depot has no demand)
    demand_compat[0, :] = 1.0
    demand_compat[:, 0] = 1.0
    
    # Combine base heuristic with demand compatibility
    # Edges that are short AND have good demand compatibility get high weight
    heuristic = base_heuristic * demand_compat
    
    # Additional Lagrangian refinement:
    # For edges where demand compatibility is low, apply a stronger penalty
    # This creates a sharper distinction between "good" and "bad" edges
    
    # Compute a dual-variable-inspired weight
    # Dual variables penalize capacity violations; we approximate this by
    # giving extra weight to edges that are "safe" (high demand_compat)
    
    # Safe edges get a boost, unsafe edges get a penalty
    safe_weight = 1.0 + 0.5 * demand_compat  # Range: [1.0, 1.5]
    unsafe_penalty = 1.0 / (1.0 + 0.5 * (1.0 - demand_compat))  # Range: [0.67, 1.0]
    
    final_weight = safe_weight * unsafe_penalty
    heuristic = heuristic * final_weight
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
