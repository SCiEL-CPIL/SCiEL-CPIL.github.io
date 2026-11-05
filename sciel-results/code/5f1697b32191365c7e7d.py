"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: we must finish well before the 120s / 150s limit.
# We track elapsed time per-call and bail out to a safe fallback if needed.
_DEADLINE_SECONDS = 115.0  # safety margin
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic that picks the next customer or returns depot (0) to close a route.
    
    Strategy: Dynamic regret-minimization / savings-based scoring.
    For each feasible candidate, compute a score that balances:
      1. Edge cost from current_node to candidate
      2. Savings from NOT going current->depot->candidate (i.e., savings heuristic)
      3. A dynamic adjustment that penalizes outlier-long edges relative to the
         average edge cost of the remaining feasible set, steering toward uniform
         segment lengths.
    
    This is a refinement of the nearest-neighbor baseline with a savings/regret
    component.
    """
    # Deadline guard
    if _elapsed() > _DEADLINE_SECONDS:
        # Fallback: nearest feasible node
        if len(unvisited_nodes) == 0:
            return int(depot)
        # Filter by capacity
        feasible = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
        if len(feasible) == 0:
            return int(depot)
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    if len(unvisited_nodes) == 0:
        return int(depot)

    # Filter to feasible nodes (capacity constraint)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible = unvisited_nodes[feasible_mask]

    if len(feasible) == 0:
        return int(depot)

    # If only one feasible, return it
    if len(feasible) == 1:
        return int(feasible[0])

    dm = distance_matrix
    cur = current_node
    dep = depot

    # Edge costs from current to each feasible node
    edge_costs = dm[cur, feasible]  # shape (n_feasible,)

    # Savings component: saving[i] = dm[cur, dep] + dm[dep, feasible[i]] - dm[cur, feasible[i]]
    # This is the classic savings heuristic value: how much we save by linking
    # cur->feasible[i] instead of cur->dep->feasible[i]
    cur_to_dep = dm[cur, dep]
    dep_to_feasible = dm[dep, feasible]
    savings = cur_to_dep + dep_to_feasible - edge_costs

    # Dynamic regret component: penalize outlier-long edges
    # Compute average edge cost of the feasible set
    mean_edge = np.mean(edge_costs)
    # Regret = how much worse this edge is compared to the mean (negative savings from being an outlier)
    # We want to AVOID outlier-long edges, so we subtract a penalty proportional to (edge_cost - mean_edge)
    # But only penalize if edge_cost > mean_edge (outlier long); don't reward outlier short too aggressively
    outlier_penalty = np.maximum(0, edge_costs - mean_edge) * 0.5  # half-weight penalty

    # Combine: we want to maximize savings while minimizing outlier penalty and edge cost
    # Score = savings - alpha * edge_costs - beta * outlier_penalty
    # Higher score = better candidate
    alpha = 0.3  # weight on direct edge cost
    beta = 0.7   # weight on outlier penalty

    scores = savings - alpha * edge_costs - beta * outlier_penalty

    # Also consider: if we're close to depot (small rest_capacity), prefer to go back
    # This prevents getting stuck with capacity but far from depot
    if rest_capacity <= 5:
        # Strongly prefer depot if we can't fit many more nodes
        # But we already filtered feasible; if feasible is non-empty, pick best
        # Add a bias toward nodes close to depot so we can return soon
        dep_bias = -0.2 * dm[feasible, dep]
        scores = scores + dep_bias

    # Select the node with the highest score
    best_idx = np.argmax(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
