<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Nearest feasible customer.
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
# EVOLVE-BLOCK-END
=======
"""Optimized CVR heuristic with deadline guard and improved scoring."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 115.0
_DEADLINE = _START_TIME + _TIMEOUT


def _elapsed():
    return time.time() - _START_TIME


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # If we're close to the deadline, just return nearest feasible
    if _elapsed() > _TIMEOUT - 5.0:
        # Simple nearest feasible fallback
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        dists = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(dists)])
    
    # Check if we can go back to depot
    if not unvisited_nodes or len(unvisited_nodes) == 0:
        return int(depot)
    
    # Filter to feasible nodes (demand <= remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    # If only one feasible node, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Compute distances from current node to feasible nodes
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Compute distance from each feasible node to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Compute minimum distance from each feasible node to other feasible nodes (regret)
    # For efficiency, use a simplified regret: min distance to other unvisited feasible nodes
    n_feasible = len(feasible_nodes)
    if n_feasible > 1:
        # For each feasible node, find min distance to other feasible nodes
        dist_matrix_feasible = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        np.fill_diagonal(dist_matrix_feasible, np.inf)
        regret = np.min(dist_matrix_feasible, axis=1)
    else:
        regret = np.zeros(n_feasible)
    
    # Scoring: combine distance, urgency (low remaining capacity after visit), and regret
    # We want to minimize total route length, so:
    # - Prefer nodes that are close (low dists)
    # - Prefer nodes that don't leave us stuck (low dist_to_depot if capacity is low)
    # - Prefer nodes with high regret (if we don't visit now, it'll be harder later)
    
    # Normalize components to [0, 1] for weighting
    max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
    min_dist = np.min(dists)
    norm_dists = (dists - min_dist) / (max_dist - min_dist + 1e-10)
    
    max_depot = np.max(dist_to_depot) if np.max(dist_to_depot) > 0 else 1.0
    min_depot = np.min(dist_to_depot)
    norm_depot = (dist_to_depot - min_depot) / (max_depot - min_depot + 1e-10)
    
    max_regret = np.max(regret) if np.max(regret) > 0 else 1.0
    min_regret = np.min(regret)
    norm_regret = (regret - min_regret) / (max_regret - min_regret + 1e-10)
    
    # Weighted score: lower is better
    # Emphasize distance most, then regret, then depot distance
    alpha = 1.0  # weight for distance
    beta = 0.5   # weight for regret  
    gamma = 0.3  # weight for depot distance (penalize if we're far from depot and capacity is low)
    
    # Adjust gamma based on remaining capacity: if capacity is low, be more careful about depot distance
    if rest_capacity < 10:
        gamma = 0.8
    elif rest_capacity < 20:
        gamma = 0.5
    
    scores = alpha * norm_dists + beta * (1.0 - norm_regret) + gamma * norm_depot
    
    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
BET: {mechanism: "Multi-criteria scoring with regret, capacity-aware depot distance weighting, and deadline guard should reduce total route length by making smarter node selection than pure nearest-neighbor", q: 0.6, delta: [0.05, 0.25]}
