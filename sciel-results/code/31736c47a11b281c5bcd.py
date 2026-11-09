"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Savings-based heuristic with angular sector smoothing:
    - Compute Clarke-Wright savings for each pair of customers.
    - Use savings to encourage edges that merge routes efficiently.
    - Add angular sector compatibility to reduce inter-sector crossings.
    - Demand-aware: prefer lower-demand nodes to pack capacity efficiently.
    - Depot edges get moderate boost for feasible returns.
    
    Returns a matrix of the same shape as distance_matrix where larger values
    indicate more attractive edges.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = distance_matrix.shape[0]
    
    # Base: strong inverse distance to favor short hops
    base = 1.0 / (distance_matrix + 1e-9) ** 2.5
    
    # Compute Clarke-Wright savings
    # savings[i, j] = d(0,i) + d(0,j) - d(i,j) for i,j >= 1
    savings = np.zeros((n, n))
    depot_to_all = distance_matrix[0, :]
    for i in range(1, n):
        for j in range(i + 1, n):
            s = depot_to_all[i] + depot_to_all[j] - distance_matrix[i, j]
            savings[i, j] = s
            savings[j, i] = s
    
    # Normalize savings to [0, 1] range for use as multiplier
    max_savings = np.max(savings)
    if max_savings > 0:
        savings_norm = savings / max_savings
    else:
        savings_norm = savings
    
    # Angular sector assignment for smooth sector compatibility
    depot = coordinates[0]
    if n > 1:
        angles = np.arctan2(coordinates[1:, 1] - depot[1], coordinates[1:, 0] - depot[0])
        # Use ~6 sectors for 50 customers
        K = 6
        sector_of = np.zeros(n, dtype=int)
        sector_of[0] = -1  # depot has no sector
        sector_of[1:] = np.floor(angles / (2 * np.pi) * K).astype(int) % K
    else:
        sector_of = np.zeros(n, dtype=int)
    
    # Smooth sector compatibility: use cosine of angle difference for smoothness
    # Instead of hard gating, use a smooth function that peaks at same sector
    sector_factor = np.ones((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                sector_factor[i, j] = 1.0
            else:
                # Compute angular distance
                angle_diff = abs(angles[i - 1] - angles[j - 1])
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                # Smooth factor: 1.0 at 0 angle diff, decays to 0.1 at pi
                # Use a cosine-based smooth function
                sector_factor[i, j] = 0.1 + 0.9 * np.cos(angle_diff / 2.0) ** 2
    
    # Demand-aware multiplier: prefer lower-demand nodes
    demand_mult = np.ones((n, n))
    for j in range(1, n):
        demand_mult[:, j] = 1.0 + 0.15 * (1.0 - demands[j] / 9.0)
    
    # Depot special handling
    depot_boost = np.ones((n, n))
    for j in range(1, n):
        depot_boost[0, j] = 1.3  # leaving depot
        depot_boost[j, 0] = 1.3  # returning to depot
    
    # Combine all factors
    # Use a weighted combination: base * (savings_boost) * sector * demand * depot
    # Savings boost: edges with high savings get extra attractiveness
    savings_boost = 1.0 + 0.5 * savings_norm
    
    result = base * savings_boost * sector_factor * demand_mult * depot_boost
    
    # Check time and ensure validity
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return a safe fallback
        result = np.maximum(result, 1e-9)
        result = np.where(np.isfinite(result), result, 1e-9)
        return result
    
    # Ensure all values are positive and finite
    result = np.maximum(result, 1e-9)
    result = np.where(np.isfinite(result), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
