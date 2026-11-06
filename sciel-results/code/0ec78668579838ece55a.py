"""Capacitated VRP heuristic: composite-score greedy with capacity-aware depot return."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 147.0  # safety margin before 150s kill
_SAFETY = 3.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick next customer or return to depot (0) to close route.
    Score: mean total route length over 50-customer instances, capacity 50, demands 1-9.
    """
    if len(unvisited_nodes) == 0:
        return 0

    # Time guard: fall back to nearest feasible if running out of time
    if _time_exceeded():
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    n_unvisited = len(unvisited_nodes)
    dm = distance_matrix

    # Filter feasible nodes (demand fits remaining capacity)
    feasible = np.array([n for n in unvisited_nodes if demands[n] <= rest_capacity], dtype=int)
    if len(feasible) == 0:
        return 0

    # If only one feasible node, take it
    if len(feasible) == 1:
        return int(feasible[0])

    # Compute distances from current to all feasible
    dist_cur = dm[current_node, feasible]

    # Distance from each feasible to depot
    dist_depot = dm[feasible, depot]

    # 2-step lookahead: for each feasible i, min distance from i to any other feasible j
    # (avoids O(n^2) for large n; with 50 nodes it's fine)
    if len(feasible) > 1:
        dist_matrix_feasible = dm[np.ix_(feasible, feasible)]
        # Set diagonal to inf so we look at other nodes
        np.fill_diagonal(dist_matrix_feasible, np.inf)
        min_next_dist = np.min(dist_matrix_feasible, axis=1)
    else:
        min_next_dist = np.zeros(len(feasible))

    # Demand of feasible nodes (higher demand = more capacity consumed = less flexibility)
    dem_feasible = demands[feasible]

    # Remaining unvisited count after picking this node
    # If we pick node i, remaining capacity = rest_capacity - dem[i]
    # Number of nodes that could still fit is a proxy for "how stranded we are"
    # Simple heuristic: if rest_capacity - dem[i] is small relative to remaining nodes,
    # we'll likely need to return to depot soon, so the depot-return cost matters more.

    # Adaptive weights based on remaining capacity and number of unvisited
    # When capacity is tight, weight depot-return more heavily
    capacity_tightness = 1.0 - (rest_capacity / 50.0)  # 0 when full, 1 when empty
    # When few nodes left, weight depot-return more
    few_nodes_factor = 1.0 if n_unvisited <= 3 else 0.5

    alpha = 0.3 + 0.4 * capacity_tightness * few_nodes_factor  # weight for depot-return
    beta = 0.2  # weight for 2-step lookahead
    gamma = 0.05 * capacity_tightness  # small penalty for high demand when capacity tight

    # Composite score: lower is better
    score = dist_cur + alpha * dist_depot + beta * min_next_dist + gamma * dem_feasible

    # Choose the node with minimum score
    best_idx = np.argmin(score)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
