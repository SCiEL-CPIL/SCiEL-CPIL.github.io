"""
Depot-pressure heuristic for ACO CVRP: models node attractiveness as a function
of demand and distance to depot, with edge weights scaled by capacity potential
and demand efficiency to prioritize coherent, well-filled routes.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Approach:
      1. Compute a "depot pressure" field: nodes with high demand are harder to serve
         and should be visited early when the truck is full.
      2. Scale customer-customer edges by the product of departure node's remaining
         capacity potential and arrival node's demand efficiency.
      3. For depot edges, use inverse distance weighted by demand urgency.
      4. Ensure positive values and finite results.
    
    Returns an n x n matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        demands = demands[:n]

    # Safe distance matrix to avoid division by zero
    dist_safe = np.maximum(distance_matrix, 1e-9)
    
    # Initialize heuristic with inverse distance (standard)
    heuristic = 1.0 / dist_safe
    
    if n > 1:
        # Extract depot distances
        d0 = distance_matrix[0, :]
        
        # Depot pressure: high-demand nodes are "urgent" to visit
        # Urgency is inversely proportional to demand (high demand = harder to fit later)
        # But also scaled by distance to depot (farther nodes are more costly to serve late)
        demand_urgency = 1.0 + (demands / capacity)  # Higher demand = more urgent
        dist_factor = d0 / np.max(d0) if np.max(d0) > 1e-9 else 1.0
        depot_pressure = demand_urgency * (1.0 + 0.5 * dist_factor)
        
        # For depot edges (row 0 and col 0):
        # Edge from depot to customer j: boosted by customer j's urgency
        # Edge from customer i to depot: boosted by customer i's urgency
        # This encourages visiting high-demand customers early (from depot) and 
        # returning to depot from high-demand customers (to reload)
        
        # Depot to customer edges (row 0)
        heuristic[0, 1:] = (1.0 / dist_safe[0, 1:]) * depot_pressure[1:]
        
        # Customer to depot edges (col 0)
        heuristic[1:, 0] = (1.0 / dist_safe[1:, 0]) * depot_pressure[1:]
        
        # For customer-customer edges:
        # Scale by departure node's remaining capacity potential and arrival node's demand efficiency
        # Remaining capacity potential: (capacity - demand[i]) / capacity
        # Demand efficiency: demand[j] / capacity
        # This prioritizes edges where the truck still has room AND the destination has meaningful demand
        
        remaining_cap_potential = (capacity - demands) / capacity  # Shape (n,)
        demand_efficiency = demands / capacity  # Shape (n,)
        
        # Edge weight boost: product of departure's remaining capacity and arrival's demand efficiency
        # This is a matrix: boost[i,j] = remaining_cap_potential[i] * demand_efficiency[j]
        edge_boost = remaining_cap_potential[:, None] * demand_efficiency[None, :]
        
        # Also incorporate Clarke-Wright savings for structural guidance
        savings = d0[:, None] + d0[None, :] - distance_matrix
        savings = np.maximum(savings, 0.0)
        
        # Normalize savings
        max_savings = np.max(savings[1:, 1:]) if n > 1 else 1.0
        if max_savings < 1e-9:
            max_savings = 1.0
        norm_savings = savings / max_savings
        
        # Combine: base inverse distance * (1 + savings boost) * (1 + edge_boost)
        # The (1 + ...) terms ensure positivity and allow the boosts to amplify
        cc_base = 1.0 / dist_safe[1:, 1:]
        cc_boost_combined = (1.0 + 0.5 * norm_savings[1:, 1:]) * (1.0 + edge_boost[1:, 1:])
        
        heuristic[1:, 1:] = cc_base * cc_boost_combined

    # Ensure all entries are strictly positive
    heuristic = np.maximum(heuristic, 1e-9)

    # Final check: no NaN or Inf
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
