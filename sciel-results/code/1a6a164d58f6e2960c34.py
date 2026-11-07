"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    This heuristic uses a demand-aware savings approach.
    The attractiveness of an edge (i, j) is determined by how much distance
    is saved by traveling directly from i to j instead of returning to the depot
    and then going to j, weighted by the demands of the nodes involved.
    This encourages the formation of routes that efficiently pack customers
    while respecting capacity constraints.
    """
    n = distance_matrix.shape[0]
    
    # Calculate the savings matrix
    # Savings(i, j) = dist(i, 0) + dist(0, j) - dist(i, j)
    # This represents the distance saved by merging the two segments i->0 and 0->j into i->j
    
    d_i_0 = distance_matrix[:, 0] # Shape (n,)
    d_0_j = distance_matrix[0, :] # Shape (n,)
    
    # Savings matrix shape (n, n)
    # s[i, j] = d_i_0[i] + d_0_j[j] - distance_matrix[i, j]
    savings = d_i_0[:, np.newaxis] + d_0_j[np.newaxis, :] - distance_matrix
    
    # Ensure savings are non-negative
    savings = np.maximum(savings, 0)
    
    # Demand-aware weighting:
    # For edge (i, j), the "value" of connecting these nodes depends on
    # how much demand is being served together. Higher demand customers
    # should be prioritized for connection to reduce the number of routes.
    # We weight the savings by the geometric mean of demands (or just demand sum).
    
    # Demand matrix: demand_weight[i, j] = demand[i] * demand[j]
    # This gives higher weight to edges between high-demand customers.
    demand_weight = demands[:, np.newaxis] * demands[np.newaxis, :]
    
    # Normalize demand weights
    max_dw = np.max(demand_weight)
    if max_dw > 0:
        norm_dw = demand_weight / max_dw
    else:
        norm_dw = np.ones_like(demand_weight)
    
    # Normalize savings to [0, 1]
    max_savings = np.max(savings)
    if max_savings > 0:
        norm_savings = savings / max_savings
    else:
        norm_savings = np.zeros_like(savings)
    
    # Base inverse distance
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Combine savings with demand-aware weighting:
    # eta[i, j] = inv_dist[i, j] * (1 + norm_savings[i, j] * (0.5 + 0.5 * norm_dw[i, j]))
    # This boosts edges with high savings AND high demand interaction.
    
    heuristic = inv_dist * (1.0 + norm_savings * (0.5 + 0.5 * norm_dw))
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
