<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Nearest feasible customer.
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select next node using savings-based heuristic.
    Prioritizes nodes that save the most distance when inserted into current route.
    """
    if not unvisited_nodes:
        return depot

    # Check if any unvisited node fits in remaining capacity
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot

    if len(feasible) == 1:
        return int(feasible[0])

    # Compute savings score for each feasible node
    # Savings = dist(current, depot) + dist(depot, node) - dist(current, node)
    # Higher savings means better to visit this node now (avoids extra depot trip)
    
    dist_current_depot = distance_matrix[current_node, depot]
    dist_depot_to_nodes = distance_matrix[depot, feasible]
    dist_current_to_nodes = distance_matrix[current_node, feasible]
    
    # Savings: how much we save by going current->node instead of current->depot->node
    savings = dist_current_depot + dist_depot_to_nodes - dist_current_to_nodes
    
    # Also consider: if we're far from depot and node is far from depot, 
    # visiting node now keeps us away from depot (good for route length)
    # But we also want to avoid getting stranded far from depot with low capacity
    
    # Capacity urgency: if rest_capacity is low, prefer nodes with lower demand
    # to leave room for more nodes
    capacity_ratio = rest_capacity / max(1.0, np.sum(demands[unvisited_nodes]))
    
    # Weighted score: primarily savings, secondarily distance from current
    # Use a soft combination
    alpha = 0.8  # Weight for savings
    beta = 0.2   # Weight for proximity (lower distance is better, so we negate)
    
    # Normalize savings and distances for fair combination
    savings_norm = savings - savings.min() + 1e-9
    dist_norm = dist_current_to_nodes - dist_current_to_nodes.min() + 1e-9
    
    # Score: high savings is good, low distance is good
    score = alpha * savings_norm - beta * dist_norm
    
    # If capacity is tight, add penalty for high-demand nodes
    if capacity_ratio < 0.3:
        demand_penalty = 0.5 * demands[feasible]
        score -= demand_penalty
    
    best_idx = np.argmax(score)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
BET: {mechanism: "Savings-based heuristic prioritizes nodes that reduce total route length by avoiding extra depot trips, outperforming nearest-neighbor which ignores global route structure", q: 0.85, delta: [0.03, 0.10]}
