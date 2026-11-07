"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ant colony optimization of CVRP.
    Uses a multi-factor heuristic: inverse distance, demand-aware scaling,
    and a mild depot-proximity bias to encourage tight routes.
    """
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    n = dm.shape[0]

    # Base: inverse distance
    base = 1.0 / (dm + 1e-9)

    # Demand-aware: scale destination attractiveness by a soft function of demand.
    # Lower demand nodes get slightly higher attractiveness (easier to pack).
    # Use a mild exponent to avoid over-penalizing.
    # demand_penalty[i] = (1 + dem[i])^(-beta), beta=0.08
    beta = 0.08
    demand_scale = (1.0 + dem) ** (-beta)

    # Depot proximity bias: edges from the depot to nearby customers are slightly
    # more attractive, encouraging short return legs.
    # Compute distance from each node to depot (node 0)
    depot_dist = dm[:, 0].copy()  # distance from each node to depot
    # Normalize to [0,1] range for soft weighting
    max_dd = np.max(depot_dist)
    if max_dd > 0:
        norm_dd = depot_dist / max_dd
    else:
        norm_dd = np.zeros(n)
    # Slight boost for nodes close to depot: factor = 1 + gamma * (1 - norm_dd)
    gamma = 0.12
    depot_bias = 1.0 + gamma * (1.0 - norm_dd)

    # Combine: base * demand_scale[dest] * depot_bias[src]
    # demand_scale applied to destination column
    # depot_bias applied to source row
    combined = base * demand_scale[np.newaxis, :] * depot_bias[:, np.newaxis]

    # Ensure positive and finite
    combined[~np.isfinite(combined)] = 1e-9
    combined[combined <= 0] = 1e-9

    return combined
# EVOLVE-BLOCK-END
