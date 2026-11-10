# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Strategy (within structural-boost family, refined):
    1. Base heuristic: 1/d (standard ACO inverse distance).
    2. Compute Clarke-Wright savings for all customer pairs.
    3. Boost customer-customer edges proportionally to their savings (smooth).
    4. Boost depot-customer edges based on the customer's max savings potential.
    5. Sanitize to positive finite values.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        new_demands = np.zeros(n, dtype=np.float64)
        min_len = min(demands.shape[0], n)
        new_demands[:min_len] = demands[:min_len]
        demands = new_demands

    dist_safe = np.maximum(distance_matrix, 1e-9)
    d0 = dist_safe[0, :]
    num_customers = n - 1

    # Base heuristic: 1/d (standard ACO)
    base_heur = 1.0 / dist_safe
    np.fill_diagonal(base_heur, 0.0)

    if num_customers < 2:
        base_heur[~np.isfinite(base_heur)] = 1e-9
        base_heur[base_heur <= 0] = 1e-9
        np.fill_diagonal(base_heur, 1e-9)
        return base_heur

    # Compute savings matrix for customer pairs
    savings_matrix = d0[1:, None] + d0[None, 1:] - dist_safe[1:, 1:]
    dem_customers = demands[1:]
    cap_constraint = (dem_customers[:, None] + dem_customers[None, :]) <= capacity
    savings_matrix[~cap_constraint] = 0.0
    np.fill_diagonal(savings_matrix, 0.0)
    savings_matrix = np.maximum(savings_matrix, 0.0)

    # Normalize savings to [0, 1]
    max_savings = np.max(savings_matrix)
    if max_savings > 0:
        norm_savings = savings_matrix / max_savings
    else:
        norm_savings = savings_matrix

    # Boost customer-customer edges
    beta_cc = 3.0
    combined = base_heur.copy()
    combined[1:, 1:] *= (1.0 + beta_cc * norm_savings)

    # Boost depot-customer edges based on max savings potential
    max_savings_per_cust = np.max(norm_savings, axis=1)
    beta_dc = 1.5
    for i in range(num_customers):
        cust_idx = i + 1
        boost = 1.0 + beta_dc * max_savings_per_cust[i]
        combined[0, cust_idx] *= boost
        combined[cust_idx, 0] *= boost

    # Sanitize
    combined[~np.isfinite(combined)] = 1e-9
    combined[combined <= 0] = 1e-9
    np.fill_diagonal(combined, 1e-9)

    return combined
# EVOLVE-BLOCK-END
