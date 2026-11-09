"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 110.0  # safety margin before 120s kill


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """Savings-based nearest neighbor with time guard."""
    # Time guard: if we're running out of time, fall back to nearest feasible
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Find feasible nodes (demand fits in remaining capacity)
        feasible_mask = demands[unvisited_nodes] <= rest_capacity
        if np.any(feasible_mask):
            feasible_nodes = unvisited_nodes[feasible_mask]
            return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_nodes[feasible_mask]
    n_feasible = len(feasible_nodes)

    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Compute scores combining:
    # 1) Distance from current node to candidate
    # 2) Savings: d(current, candidate) - d(current, depot) - d(candidate, depot) + d(depot, candidate)
    #    Actually savings for inserting candidate: d(depot,curr) + d(cand,depot) - d(curr,cand)
    #    Higher savings = more beneficial to visit
    # We want to minimize distance but also prefer high savings
    # Score = dist_from_current - alpha * savings (lower is better)

    dists = distance_matrix[current_node, feasible_nodes]  # (n_feasible,)
    dist_to_depot = distance_matrix[feasible_nodes, depot]  # (n_feasible,)
    dist_depot_to_curr = distance_matrix[depot, current_node]  # scalar

    # Savings of visiting candidate i instead of going back to depot:
    # saving = d(depot, current) + d(i, depot) - d(current, i)
    # Positive saving means it's beneficial to visit i before returning to depot
    savings = dist_depot_to_curr + dist_to_depot - dists

    # Weighted score: primarily distance, secondarily savings (prefer higher savings)
    # Normalize to make them comparable
    max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
    max_saving = np.max(savings) if np.max(savings) > 0 else 1.0

    alpha = 0.3  # weight for savings component
    scores = dists / max_dist - alpha * savings / max_saving

    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])

# EVOLVE-BLOCK-END
