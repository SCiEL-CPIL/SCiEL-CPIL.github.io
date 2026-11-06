"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return depot if we've exceeded 145s
    elapsed = time.time() - _start_time
    if elapsed > 145:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited = np.array(unvisited_nodes, dtype=int)
    n = len(unvisited)
    
    if n == 0:
        return depot
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited]
    
    # Get demands of unvisited nodes
    unvisited_demands = demands[unvisited]
    
    # Feasibility: demand must fit in remaining capacity
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_nodes = unvisited[feasible_indices]
    feasible_dist = dist_from_current[feasible_indices]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # Clarke-Wright savings heuristic approach:
    # For each feasible candidate node j, compute a score that balances:
    # 1. Distance from current to j (lower is better)
    # 2. Distance from j to depot (lower is better, since we'll eventually go to depot)
    # 3. The "savings" of visiting j now vs. saving it for a later route
    
    # Compute distance from each feasible candidate to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Savings heuristic: visiting j saves (dist[current->depot] + dist[depot->j] - dist[current->j])
    # But we're already at current, so the relevant comparison is:
    # If we go current->j->depot vs current->depot (and j gets visited in another route)
    # The savings of including j in this route is:
    #   dist[current->depot] + dist[depot->j] - dist[current->j] - dist[j->depot]
    # Wait, that's not quite right. Let me think more carefully.
    
    # Actually, for a greedy construction:
    # We want to pick j that minimizes the total route length.
    # A good proxy: minimize dist[current->j] + dist[j->depot] (the cost of adding j and going to depot)
    # But we also want to consider that not visiting j now might mean it's visited in another route.
    
    # Let's use a combined score:
    # score_j = dist[current->j] + alpha * dist[j->depot]
    # where alpha weights the importance of going back to depot.
    
    # Actually, a better approach inspired by Clarke-Wright:
    # The savings of concatenating two routes (current->...->j and j->...->depot) 
    # vs keeping them separate is:
    #   s(j) = dist[current->depot] + dist[depot->j] - dist[current->j]
    # But since we're building a single route from current, we want to maximize savings.
    # However, we also need to consider the demand and remaining capacity.
    
    # Let's use a weighted score:
    # score_j = w1 * dist[current->j] + w2 * dist[j->depot]
    # Lower score is better.
    
    # For CVRP with capacity constraints, a good heuristic is:
    # 1. Prefer nodes that are close to current (low dist[current->j])
    # 2. Prefer nodes that are close to depot (low dist[j->depot]) - this helps close the route efficiently
    # 3. Consider the "angle" or direction - nodes in the same general direction as depot are preferred
    
    # Compute the combined score
    # Use a soft combination that balances proximity to current and proximity to depot
    alpha = 0.3  # Weight for distance to depot
    
    scores = feasible_dist + alpha * dist_to_depot
    
    # Additionally, penalize nodes that are "far" in terms of potential future savings
    # A node that is far from both current and depot is less attractive
    
    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
