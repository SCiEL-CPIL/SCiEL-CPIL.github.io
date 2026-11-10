"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Depot-pressure field heuristic for ACO CVRP.
    
    Design:
    - Base: inverse distance (closer is more attractive)
    - Depot-pressure: nodes with higher demand are "harder" to fit late in a route,
      so we boost edges that lead toward high-demand nodes early (when capacity
      is available). This is modeled by scaling edge (i,j) by a factor that
      increases with demand[j] (the arrival node) and decreases with the
      cumulative demand that would have been loaded before reaching j.
    - Since we don't know the route order a priori, we approximate by using
      the node's demand as a proxy: higher demand at arrival means we want
      to reach it when the truck is fuller, i.e., earlier in the sequence.
    - We also add a mild clustering bonus: edges between nearby customers
      get a small boost to encourage compact routes.
    
    Structural change from Clarke-Wright savings: instead of computing pairwise
    savings (which is a global optimization signal), we model a local pressure
    field based on demand and distance. This is a different design family:
    a potential-field approach where each node exerts a "pull" proportional
    to its demand, and edges are weighted by how well they align with this field.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Base: inverse distance
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Demand efficiency: higher demand nodes are more "valuable" to visit early
    # Normalize demands to [0, 1] range
    max_demand = demands.max() if demands.max() > 0 else 1.0
    demand_norm = demands / max_demand
    
    # Depot-pressure field: for each edge (i, j), the attractiveness is boosted
    # if j has high demand (we want to serve it early) and i is close to the depot
    # (suggesting we're still early in the route)
    # Approximation: scale by demand_norm[j] * (1 / (1 + dist(0,i)/mean_dist))
    mean_dist_to_depot = np.mean(distance_matrix[0, 1:n]) if n > 1 else 1.0
    
    # Create depot proximity factor for each node (higher = closer to depot)
    depot_proximity = 1.0 / (1.0 + distance_matrix[0, :] / mean_dist_to_depot)
    
    # For edge (i, j): boost = demand_norm[j] * depot_proximity[i]
    # This means: if we're at a node close to the depot (early in route) and 
    # heading to a high-demand node, that edge is more attractive
    demand_boost = np.outer(depot_proximity, demand_norm)
    
    # Clustering bonus: edges between nearby customers get a small boost
    # Use a Gaussian kernel on distance
    dist_matrix = distance_matrix.copy()
    np.fill_diagonal(dist_matrix, 0)
    sigma = 0.1  # Bandwidth for clustering
    cluster_bonus = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    
    # Combine: base * (1 + alpha * demand_boost + beta * cluster_bonus)
    # Use moderate coefficients to avoid overwhelming the distance signal
    alpha = 0.5
    beta = 0.3
    
    heuristic = base * (1.0 + alpha * demand_boost + beta * cluster_bonus)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen for 50x50 matrix, but guard just in case
        pass
    
    return heuristic
# EVOLVE-BLOCK-END
