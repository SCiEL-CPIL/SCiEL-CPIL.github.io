"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Capacitated VRP next-node heuristic.
    Returns 0 (depot) to close route, or a customer index.
    """
    if _time_exceeded():
        # Fast fallback: nearest feasible
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        if len(feasible) == 1:
            return int(feasible[0])
        dists = distance_matrix[current_node, feasible]
        return int(feasible[int(np.argmin(dists))])

    # Filter feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]

    if not feasible:
        return int(depot)

    if len(feasible) == 1:
        return int(feasible[0])

    feasible_arr = np.array(feasible)
    n = len(feasible_arr)
    demands_feas = np.array([demands[i] for i in feasible_arr])

    # Immediate distance from current node to each candidate
    immediate = distance_matrix[current_node, feasible_arr]

    # Distance matrix among feasible candidates and depot
    # Build: for each candidate, distances to all feasible + depot
    all_targets = np.concatenate([feasible_arr, [depot]])
    dist_to_all = distance_matrix[feasible_arr][:, all_targets]  # shape (n, n+1)

    # --- 1-step lookahead: min distance from candidate to any other feasible or depot ---
    look1 = dist_to_all.copy()
    for i in range(n):
        look1[i, i] = np.inf
    lookahead_1 = np.min(look1, axis=1)

    # --- 2-step lookahead: for each candidate i, estimate min over j != i of
    #     dist(i, j) + min over k != j of dist(j, k)   (where k is feasible or depot)
    # This approximates the cost of the next two moves after visiting i.
    # For efficiency, precompute for each node j: min_dist_from_j = min over k != j of dist(j,k)
    # Then 2-step cost for candidate i = min over j != i of dist(i,j) + min_dist_from_j
    # (excluding j=i from the inner min is already handled by min_dist_from_j)

    # min_dist_from_j: for each feasible node j, min distance to any other feasible or depot
    min_dist_from_j = np.empty(n)
    for j in range(n):
        row = dist_to_all[j].copy()
        row[j] = np.inf  # exclude self
        min_dist_from_j[j] = np.min(row)

    # 2-step lookahead for candidate i:
    # min over j != i of dist(i, j) + min_dist_from_j[j]
    # Build matrix: dist_to_all[:, :n] is (n, n), add min_dist_from_j to each column
    two_step_matrix = dist_to_all[:, :n] + min_dist_from_j[np.newaxis, :]
    # Exclude self: set diagonal to inf
    for i in range(n):
        two_step_matrix[i, i] = np.inf
    lookahead_2 = np.min(two_step_matrix, axis=1)

    # --- Capacity urgency: prefer nodes with higher demand relative to rest_capacity ---
    # If rest_capacity is small, high-demand nodes become infeasible quickly.
    # urgency = demand / rest_capacity (higher = more urgent)
    if rest_capacity > 0:
        urgency = demands_feas / rest_capacity
    else:
        urgency = np.zeros(n)

    # --- Composite score ---
    # We want to minimize: immediate + alpha * lookahead_1 + beta * lookahead_2 - gamma * urgency
    # Negative urgency term means we prefer higher-urgency nodes.
    alpha = 0.4
    beta = 0.3
    gamma = 1.5  # weight for urgency (subtract, so higher urgency = lower score)

    scores = immediate + alpha * lookahead_1 + beta * lookahead_2 - gamma * urgency

    best_idx = int(np.argmin(scores))
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
