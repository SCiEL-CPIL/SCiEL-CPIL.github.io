"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # wall-clock safety margin (120s limit + buffer)
_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = len(unvisited_nodes)
    if n == 0:
        return depot

    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: nearest feasible
        feasible = [i for i in range(n) if demands[unvisited_nodes[i]] <= rest_capacity]
        if not feasible:
            return depot
        d = distance_matrix[current_node, unvisited_nodes[feasible]]
        return int(unvisited_nodes[feasible[np.argmin(d)]])

    # Filter feasible nodes (demand fits remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited_nodes[feasible_mask]
    nf = len(feasible_nodes)

    # Distance from current node to each feasible candidate
    dist_from_current = distance_matrix[current_node, feasible_nodes]

    # Distance from depot to each feasible candidate (for savings)
    dist_from_depot = distance_matrix[depot, feasible_nodes]

    # Clarke-Wright savings for each candidate:
    # savings_i = dist(depot, i) + dist(current, i) - dist(depot, current)
    # Actually the standard savings when inserting i between depot and current:
    # But here we're extending a route, so the "saving" of visiting i now vs. later
    # is related to avoiding a separate trip. We use:
    # score_i = dist(depot, i) + dist_from_current[i] - dist(depot, current)
    # Lower savings means less benefit from not going to depot... 
    # Actually let's think differently:
    # If we go to i now, the incremental cost is dist_from_current[i].
    # If we skip i and go to depot, then later do a new route to i, cost is dist(depot, current) + dist(depot, i).
    # So "savings" of visiting i now = dist(depot, current) + dist(depot, i) - dist_from_current[i]
    # Higher savings = better to visit now.
    dist_depot_current = distance_matrix[depot, current_node]
    savings = dist_depot_current + dist_from_depot - dist_from_current

    # Regret: for each feasible candidate i, compute how much worse the second-best
    # option is compared to the best option for "covering" i's region.
    # A simpler regret: for candidate i, look at all other feasible candidates j.
    # The regret of NOT picking i is: min over j of dist(i, j) - dist_from_current[i]
    # No, let's use a standard regret-based approach:
    # For each candidate i, regret_i = second_best_score - best_score among alternatives
    # where "score" is the distance from current node.
    # But that's just nearest-neighbor with regret.
    
    # Better regret: For each candidate i, consider the cost if we don't pick i now.
    # The "regret" is how much more expensive it becomes to visit i later.
    # Approximate: for candidate i, the regret is the difference between the best
    # connection to i from current vs. the second best, but that doesn't make sense
    # since we're choosing which node to visit next.
    
    # Let's use a different regret formulation:
    # For each candidate i, compute its "urgency" based on how isolated it is.
    # If i is far from all other feasible nodes, it's urgent (regret of delaying is high).
    # regret_i = min_j (dist(i, j)) for j != i in feasible set
    # But we want to prioritize nodes that are uniquely well-positioned.
    
    # Actually, let's use the savings-based score with a regret adjustment:
    # score_i = alpha * savings_i + beta * regret_i
    # where regret_i captures how much we lose by not picking i.
    
    # Compute pairwise distances among feasible nodes for regret
    if nf > 1:
        # For each feasible node i, find its nearest other feasible node
        # dist_matrix among feasible nodes
        pair_dists = np.zeros((nf, nf))
        for idx_i in range(nf):
            pair_dists[idx_i] = distance_matrix[feasible_nodes[idx_i], feasible_nodes]
        
        # Regret for node i: if we don't pick i, the best alternative to cover i's area
        # is the nearest other node j. The "cost" of that alternative is dist_from_current[j] + dist(j, i).
        # The "cost" of picking i now is dist_from_current[i].
        # Regret_i = min_j (dist_from_current[j] + dist(j, i)) - dist_from_current[i]
        # If regret is negative, picking i now is cheaper than any alternative.
        # If regret is positive, there's a cheaper way to cover i's area.
        
        # For each i, compute min over j != i of (dist_from_current[j] + dist(j, i))
        # We can vectorize this
        # cost_via_j[i, j] = dist_from_current[j] + pair_dists[i, j]
        cost_via = dist_from_current[None, :] + pair_dists  # shape (nf, nf)
        # Set diagonal to inf (can't go via self)
        np.fill_diagonal(cost_via, np.inf)
        min_cost_via = np.min(cost_via, axis=1)  # shape (nf,)
        
        # Regret: how much more expensive is the best alternative vs. direct
        regret = min_cost_via - dist_from_current
        
    else:
        regret = np.array([0.0])
    
    # Combine: higher savings and higher regret are both good
    # Normalize to avoid scale issues
    savings_range = savings.max() - savings.min() if nf > 1 else 1.0
    regret_range = regret.max() - regret.min() if nf > 1 else 1.0
    
    if savings_range < 1e-9:
        savings_norm = np.zeros(nf)
    else:
        savings_norm = (savings - savings.min()) / savings_range
    
    if regret_range < 1e-9:
        regret_norm = np.zeros(nf)
    else:
        regret_norm = (regret - regret.min()) / regret_range
    
    # Weighted combination
    alpha = 0.6  # weight for savings
    beta = 0.4   # weight for regret
    
    combined = alpha * savings_norm + beta * regret_norm
    
    # Pick the candidate with highest combined score
    best_idx = np.argmax(combined)
    return int(feasible_nodes[best_idx])

# EVOLVE-BLOCK-END
