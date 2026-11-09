"""Capacitated VRP next-node heuristic: Savings with Centroid Bias and Capacity Penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_MODULE_START = time.time()
TIME_LIMIT = 110.0  # seconds; safety margin below 120s/150s limits


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP next-node selection.
    
    Strategy: 
    1. Clarke-Wright Savings: Minimize d(current, j) - d(depot, j) to maximize savings.
    2. Centroid Bias: Penalize distance from candidate j to the centroid of all unvisited nodes.
       This encourages the route to move toward the center of remaining demand, reducing
       backtracking and improving cluster cohesion.
    3. Capacity Penalty: As capacity tightens, increase penalty for distance to depot
       to ensure we can return to depot without exceeding capacity or traveling too far.
    """
    # Deadline guard
    elapsed = time.time() - _MODULE_START
    if elapsed > TIME_LIMIT:
        if len(unvisited_nodes) == 0:
            return 0
        feasible = [u for u in unvisited_nodes if demands[u] <= rest_capacity]
        if not feasible:
            return 0
        # Fallback: Nearest feasible node
        return int(min(feasible, key=lambda u: distance_matrix[current_node, u]))

    if len(unvisited_nodes) == 0:
        return 0

    # Filter by capacity feasibility
    unvisited_arr = np.asarray(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)

    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Distances from current node to all feasible nodes
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Distances from each feasible node to depot
    dists_to_depot = distance_matrix[feasible_nodes, depot]

    # --- Structural Change: Centroid Bias ---
    # Calculate the "centroid" of the remaining unvisited nodes in the distance space.
    # Since we don't have coordinates, we approximate the "center" by finding the node
    # that minimizes the sum of distances to all other unvisited nodes (1-median approximation)
    # OR simply use the mean of the distance vectors if we had them. 
    # Without coordinates, a good proxy for "center of cluster" is the node with the
    # smallest sum of distances to all other unvisited nodes.
    
    # However, computing the 1-median for every step is O(N^2) which is fine for N=50.
    # Let's find the node in unvisited_nodes that is closest to all others (median node).
    # This node represents the "core" of the remaining cluster.
    
    if len(unvisited_arr) > 1:
        # Compute sum of distances from each unvisited node to all other unvisited nodes
        # dist_matrix[unvisited_arr, unvisited_arr] gives pairwise distances
        pairwise = distance_matrix[unvisited_arr, unvisited_arr]
        sum_dists = np.sum(pairwise, axis=1)
        median_node_idx = np.argmin(sum_dists)
        median_node = unvisited_arr[median_node_idx]
        
        # Distance from each feasible candidate to this median node
        dists_to_median = distance_matrix[feasible_nodes, median_node]
    else:
        # If only one unvisited node, it is the median
        median_node = unvisited_arr[0]
        dists_to_median = distance_matrix[feasible_nodes, median_node]

    # --- Capacity Tightness ---
    total_demand = float(np.sum(demands[unvisited_arr]))
    if total_demand > 0:
        cap_ratio = rest_capacity / total_demand
    else:
        cap_ratio = 1.0
    cap_ratio = min(1.0, max(0.0, cap_ratio))

    # Weights
    # w_dep: Penalty for being far from depot, increases as capacity tightens.
    # w_med: Weight for centroid bias, encourages moving toward the cluster core.
    w_dep = 0.5 * (1.0 - cap_ratio)
    w_med = 0.2
    
    # Composite Score:
    # Primary: Savings (d_cur - d_depot)
    # Secondary: Distance to median (d_med) - encourages cohesion
    # Tertiary: Depot penalty (w_dep * d_depot) - ensures returnability
    #
    # score = (d_cur - d_depot) + w_med * d_med + w_dep * d_depot
    # score = d_cur + (w_dep - 1) * d_depot + w_med * d_med
    
    scores = dists_from_current + (w_dep - 1.0) * dists_to_depot + w_med * dists_to_median

    best_idx = int(np.argmin(scores))
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
