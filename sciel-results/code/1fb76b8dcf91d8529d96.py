"""CVRP Heuristic: Full-route simulation with capacity-tightness penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0
_DEADLINE = None


def _simulate_route_completion(start_node, remaining_node_set, remaining_cap, demands, dist_mat, depot):
    """
    Greedily complete a route starting from start_node using nearest-neighbor
    with capacity feasibility. Returns total additional distance (not including
    the edge that brought us to start_node).
    """
    total_dist = 0.0
    current = start_node
    remaining_set = set(remaining_node_set)
    cap = remaining_cap

    while remaining_set:
        # Find feasible nodes
        feasible = [n for n in remaining_set if demands[n] <= cap]
        if not feasible:
            break
        # Pick nearest feasible
        best_node = min(feasible, key=lambda n: dist_mat[current, n])
        total_dist += dist_mat[current, best_node]
        cap -= demands[best_node]
        remaining_set.remove(best_node)
        current = best_node

    # Return to depot
    total_dist += dist_mat[current, depot]
    return total_dist


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node by simulating full route completion for each candidate,
    using nearest-neighbor with capacity checks. A capacity-tightness penalty
    discourages choices that leave little slack, reducing dead-ends.
    """
    global _start_time, _DEADLINE

    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT

    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)

    # Convert unvisited_nodes to numpy array
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)

    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)

    # Find feasible nodes
    feasible_mask = demands[unvisited_arr] <= rest_capacity

    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)

    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)

    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])

    # For each candidate J, simulate completing the route after going to J
    # Cost = d(current, J) + simulated_completion(J, rest_cap - demand[J], unvisited - {J})
    # Plus a capacity-tightness penalty: penalize if remaining capacity after J is low
    # relative to total remaining demand (encourages not wasting capacity on detours)

    # Baseline: cost of closing route now (go back to depot)
    baseline_cost = float(distance_matrix[current_node, depot])

    scores = np.empty(n_feasible, dtype=np.float64)
    demands_arr = np.asarray(demands, dtype=np.float64)

    for i in range(n_feasible):
        j = int(feasible_nodes[i])
        d_cj = float(distance_matrix[current_node, j])
        cap_after_j = rest_capacity - float(demands_arr[j])

        # Simulate completion from J with remaining capacity
        remaining_nodes = [int(n) for n in unvisited_arr if int(n) != j]
        sim_dist = _simulate_route_completion(
            j, remaining_nodes, cap_after_j,
            demands_arr, distance_matrix, depot
        )

        total_cost = d_cj + sim_dist

        # Capacity-tightness penalty:
        # If cap_after_j is very small relative to remaining demand, we risk
        # being stuck. Penalize proportionally to (total_remaining_demand - cap_after_j)
        # normalized by total_remaining_demand.
        total_remaining_demand = float(np.sum(demands_arr[unvisited_arr]))
        if total_remaining_demand > 0:
            tightness = max(0.0, 1.0 - cap_after_j / max(total_remaining_demand, 1.0))
            # Scale penalty: if tightness > 0.5, start penalizing
            penalty = 0.0
            if tightness > 0.5:
                penalty = 2.0 * (tightness - 0.5) * total_cost
        else:
            penalty = 0.0

        scores[i] = total_cost + penalty

    # Choose the candidate with minimum total cost
    best_idx = int(np.argmin(scores))
    best_cost = scores[best_idx]

    # If the best candidate's cost is worse than just closing the route, close it
    if best_cost > baseline_cost:
        return int(depot)

    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
