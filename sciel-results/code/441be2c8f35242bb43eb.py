"""
Structural change: Depot-Pressure Field Heuristic for ACO CVRP.

Instead of distance-based inverse heuristic + greedy reference edges,
we model a "depot pressure" field where:
- High-demand nodes have higher "pressure" (harder to serve, should be visited early)
- Edge attractiveness scales with the departure node's remaining capacity potential
  and the arrival node's demand efficiency
- This encourages routes that cluster high-demand nodes early when the truck is full,
  leading to more compact routes and fewer trips

The heuristic is computed as:
  eta[i,j] = (1 / d[i,j]) * (1 + alpha * demand[j] / capacity) * (1 + beta * (1 - demand[i]/capacity))

This creates a bias towards:
- Short edges (1/d)
- Visiting high-demand nodes (demand[j] term)
- Departing from nodes that have already consumed some capacity (demand[i] term)
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    """
    deadline = time.time() + 290.0
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n == 0:
        return np.array([])
        
    if d.shape != (n, n):
        return 1.0 / (d + 1e-9)
        
    eps = 1e-9
    
    # Base inverse distance
    inv_dist = 1.0 / (d + eps)
    
    # Demand-aware scaling
    # alpha: weight for arrival node's demand (higher demand = more attractive to visit)
    # beta: weight for departure node's capacity consumption (higher consumption = more attractive to depart from)
    alpha = 0.5
    beta = 0.3
    
    # Normalize demands to [0, 1] range
    demands_norm = np.asarray(demands, dtype=np.float64) / capacity
    
    # Create demand pressure fields
    # arrival_pressure[j] = 1 + alpha * demand_norm[j]
    # departure_pressure[i] = 1 + beta * demand_norm[i]
    
    arrival_pressure = 1.0 + alpha * demands_norm  # shape (n,)
    departure_pressure = 1.0 + beta * demands_norm  # shape (n,)
    
    # Outer product to create edge-wise pressure
    # eta[i,j] = inv_dist[i,j] * departure_pressure[i] * arrival_pressure[j]
    departure_matrix = departure_pressure[:, np.newaxis]  # (n, 1)
    arrival_matrix = arrival_pressure[np.newaxis, :]  # (1, n)
    
    eta = inv_dist * departure_matrix * arrival_matrix
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(eta, 0.0)
    
    # Normalize to mean 1.0 for off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    mean_eta = np.mean(eta[mask])
    
    if mean_eta > 0 and np.isfinite(mean_eta):
        eta = eta / mean_eta
    else:
        eta = inv_dist.copy()
        np.fill_diagonal(eta, 0.0)
    
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
