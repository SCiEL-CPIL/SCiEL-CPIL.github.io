"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Structural approach: angular-sector clustering heuristic.
    - Assign each customer to an angular sector (corridor) from the depot.
    - Strongly encourage intra-sector edges, strongly discourage inter-sector edges.
    - Within a sector, use inverse distance with high exponent for local packing.
    - Depot edges get a moderate boost to ensure return-to-depot is feasible.
    - Demand-aware: slightly prefer lower-demand nodes to fill capacity efficiently.
    
    Returns a matrix of the same shape as distance_matrix where larger values
    indicate more attractive edges.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = distance_matrix.shape[0]
    
    # Base: strong inverse distance to favor short hops
    base = 1.0 / (distance_matrix + 1e-9) ** 3.0
    
    # Angular sector assignment: divide [0, 2*pi) into K sectors
    depot = coordinates[0]
    if n > 1:
        angles = np.arctan2(coordinates[1:, 1] - depot[1], coordinates[1:, 0] - depot[0])
        # Use ~8 sectors for 50 customers (~6-7 per sector, good for capacity 50)
        K = 8
        sector_of = np.zeros(n, dtype=int)
        sector_of[0] = -1  # depot has no sector
        # Assign customers to sectors
        sector_of[1:] = np.floor(angles / (2 * np.pi) * K).astype(int) % K
    else:
        sector_of = np.zeros(n, dtype=int)
    
    # Sector compatibility matrix: 1 if same sector (or depot), 0 otherwise
    # Depot connects to all sectors
    sector_compat = np.ones((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if sector_of[i] != sector_of[j]:
                sector_compat[i, j] = 0.0
    
    # Apply sector gating: inter-sector edges get a very strong penalty
    # Use a multiplicative factor: same sector = 1.0, different sector = 0.05
    sector_factor = np.where(sector_compat == 1.0, 1.0, 0.05)
    
    # Demand-aware multiplier: prefer lower-demand nodes slightly
    # This helps pack more customers per route
    demand_mult = np.ones((n, n))
    for j in range(1, n):
        # Lower demand -> higher attractiveness (mild effect)
        demand_mult[:, j] = 1.0 + 0.1 * (1.0 - demands[j] / 9.0)
    
    # Depot special handling: depot edges should be moderately attractive
    # so ants can return to depot when route is full
    # Boost depot->customer and customer->depot edges slightly
    depot_boost = np.ones((n, n))
    for j in range(1, n):
        depot_boost[0, j] = 1.2  # leaving depot
        depot_boost[j, 0] = 1.2  # returning to depot
    
    # Combine
    result = base * sector_factor * demand_mult * depot_boost
    
    # Ensure all values are positive and finite
    result = np.maximum(result, 1e-9)
    result = np.where(np.isfinite(result), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
