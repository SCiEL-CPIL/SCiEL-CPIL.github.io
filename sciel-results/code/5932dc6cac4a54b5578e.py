"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """Savings + 2-step lookahead heuristic with time guard."""
    if len(unvisited_nodes) == 0:
        return 0

    # Time guard: if we're running out of time, fall back to nearest feasible
    if time.time() - _START_TIME > _TIME_LIMIT:
        feasible_mask = demands[unvisited_nodes] <= rest_capacity
        if np.any(feasible_mask):
            feasible_nodes = unvisited_nodes[feasible_mask]
            return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])
        return 0

    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_nodes[feasible_mask]
    n_feasible = len(feasible_nodes)

    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Base scores: distance from current + savings consideration
    dists = distance_matrix[current_node, feasible_nodes]  # (n_feasible,)
    dist_to_depot = distance_matrix[feasible_nodes, depot]  # (n_feasible,)
    dist_depot_to_curr = distance_matrix[depot, current_node]  # scalar

    # Savings of visiting candidate i instead of going back to depot
    savings = dist_depot_to_curr + dist_to_depot - dists

    # 2-step lookahead: for each candidate, estimate the best next move cost
    # This helps avoid picking a node that leads to a bad continuation
    # For each candidate i, find the min distance from i to any other feasible node
    # or to the depot (whichever gives better total route)
    
    # Cost if we go to candidate i and then back to depot:
    cost_return_depot = dists + dist_to_depot
    
    # Cost if we go to candidate i and then to the best other feasible node:
    # For efficiency, compute pairwise distances among feasible nodes
    if n_feasible <= 50:  # small enough for pairwise
        # Distance from each feasible node to all other feasible nodes
        # dist_matrix_feasible[i, j] = distance from feasible_nodes[i] to feasible_nodes[j]
        dists_between = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]  # (n, n)
        
        # For each candidate i, the best continuation is min over j != i of:
        #   d(i, j) + d(j, depot)  (if j's demand fits after i)
        # Simplification: just use min d(i, j) + d(j, depot) for all j != i
        # We'll approximate by finding min over j != i
        
        # For each i, find best j != i
        best_continuation = np.zeros(n_feasible)
        for i in range(n_feasible):
            # Exclude self
            others = np.arange(n_feasible) != i
            if np.any(others):
                # Cost = d(i, j) + d(j, depot) for each j
                cont_costs = dists_between[i, others] + dist_to_depot[others]
                best_continuation[i] = np.min(cont_costs)
            else:
                best_continuation[i] = dist_to_depot[i]
        
        # Total cost if we pick i: d(current, i) + best_continuation[i]
        total_costs = dists + best_continuation
    else:
        # For larger sets, use a simpler approximation
        total_costs = dists + dist_to_depot

    # Combine with savings: prefer nodes with higher savings
    # Score = total_cost - beta * savings
    max_cost = np.max(total_costs) if np.max(total_costs) > 0 else 1.0
    max_saving = np.max(savings) if np.max(savings) > 0 else 1.0
    
    beta = 0.2  # weight for savings
    scores = total_costs / max_cost - beta * savings / max_saving

    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])

# EVOLVE-BLOCK-END
