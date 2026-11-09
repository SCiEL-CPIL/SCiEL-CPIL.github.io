"""Optimized heuristics for ACO CVRP with instance-normalized inverse distance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Strategy:
    - Base: Inverse distance with exponent 1.5 for strong discrimination.
    - Refinement: Instance-wise normalization so mean off-diagonal is ~1.0,
      balancing initial pheromone (1.0) with heuristic across instances.
    - Demand-aware soft penalty: mildly reduce attractiveness of edges between
      high-demand customers to encourage load balancing in routes.
    """
    deadline = time.time() + 290.0

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    if n == 0:
        return np.array([])

    if d.shape != (n, n):
        return 1.0 / (d + 1e-9)

    eps = 1e-9

    # Base heuristic: Inverse distance with exponent 1.5
    # Stronger discrimination toward closer nodes
    eta = 1.0 / ((d + eps) ** 1.5)

    # Demand-aware adjustment:
    # Penalize edges between high-demand customers to spread load
    dem = np.asarray(demands, dtype=np.float64)
    if dem.shape[0] >= n:
        dem = dem[:n]
    else:
        # Pad if necessary (should not happen per contract)
        padded = np.zeros(n, dtype=np.float64)
        padded[:dem.shape[0]] = dem
        dem = padded

    max_dem = np.max(dem)
    if max_dem > 0:
        norm_dem = dem / max_dem
        dem_prod = np.outer(norm_dem, norm_dem)
        # Small penalty: at most ~8% reduction for highest-demand pairs
        penalty_factor = 0.08
        adjustment = 1.0 - penalty_factor * dem_prod
        eta = eta * adjustment

    # Normalize the heuristic matrix
    # Mean of off-diagonal elements should be approximately 1.0
    mask = ~np.eye(n, dtype=bool)
    mean_eta = np.mean(eta[mask])

    if mean_eta > 0 and np.isfinite(mean_eta):
        eta = eta / mean_eta
    else:
        eta = 1.0 / ((d + eps) ** 1.5)

    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9

    return eta
# EVOLVE-BLOCK-END
