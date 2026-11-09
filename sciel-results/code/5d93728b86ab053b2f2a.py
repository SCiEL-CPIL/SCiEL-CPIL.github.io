# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node to visit using a savings-based heuristic with
    capacity feasibility and a structural scoring that favors compact routes.
    Returns depot (0) to close route if no feasible customer or if closing is better.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return int(depot)
    
    unvisited_nodes = np.array(unvisited_nodes, dtype=int)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    n = len(distance_matrix)
    
    # Find feasible customers (those whose demand fits in remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    # Compute distances from current node to all feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Compute distances from depot to all feasible nodes
    dist_to_depot = distance_matrix[depot, feasible_nodes]
    
    # Savings-based score: 
    # If we visit node i, the "savings" compared to serving it separately from depot
    # is: dist(current, depot) - dist(current, i) - dist(i, depot)
    # But since we're already on a route, a better proxy is:
    # We want to minimize the marginal cost of adding this node.
    # Marginal cost ≈ dist(current, i) + dist(i, depot) - dist(current, depot)
    # Minimizing marginal cost = maximizing savings = dist(current, depot) - dist(current, i) - dist(i, depot)
    # Since dist(current, depot) is constant for all i, we maximize: -dist(current, i) - dist(i, depot)
    # i.e., minimize dist(current, i) + dist(i, depot)
    
    # However, a pure nearest-neighbor on dist(current,i) can be suboptimal.
    # Let's use a weighted combination: prioritize nodes that are close to current
    # AND close to depot (compact), but also consider the angle/geometry.
    
    # Structural change: Use a "regret" or "savings" based scoring
    # For each feasible node i, compute:
    #   score_i = dist(current, i) + alpha * dist(i, depot)
    # where alpha balances proximity to current vs. proximity to depot.
    # A lower score is better.
    
    # Additionally, we can add a penalty for nodes that are "far" in the sense
    # that they would force a long detour. The classic savings heuristic suggests
    # that the saving from linking i and j on the same route is:
    #   s(i,j) = d(depot,i) + d(depot,j) - d(i,j)
    # When extending a route from current to i, the "saving" relative to closing
    # the route and starting a new one for i is related to how much we save by
    # not going back to depot now.
    
    # Let's use a composite score:
    #   cost_i = dist(current, i) + beta * dist(i, depot)
    # This naturally prefers nodes that are close to both current and depot.
    
    beta = 0.3  # Weight for depot proximity
    
    composite_score = dist_from_current + beta * dist_to_depot
    
    # Additionally, consider the "angle" or direction: nodes that are in the
    # general direction of other unvisited nodes may be better. But this is complex.
    # Let's stick with the composite score for now.
    
    # Also, we should consider capacity: nodes with larger demands might be better
    # to visit earlier while capacity is high. Add a small penalty for high demand
    # to encourage visiting large-demand nodes earlier.
    demand_penalty = 0.01 * demands[feasible_nodes]
    
    final_score = composite_score + demand_penalty
    
    # Pick the node with the minimum final score
    best_idx = np.argmin(final_score)
    best_node = feasible_nodes[best_idx]
    
    # Check if closing the route is better than continuing
    # If the best node is very far, it might be better to close the route
    # A simple heuristic: if dist(current, best_node) > dist(current, depot) * 1.5,
    # consider closing. But this is too aggressive.
    
    # Let's just return the best feasible node
    return int(best_node)
# EVOLVE-BLOCK-END
