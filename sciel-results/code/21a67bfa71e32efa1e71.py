"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic to pick the next customer or return to depot (0).
    
    Strategy:
    1. Filter unvisited nodes by capacity feasibility.
    2. If no feasible nodes, return to depot.
    3. Score each feasible node using a combination of:
       - Immediate distance from current node
       - Distance from that node back to depot (to penalize nodes far from depot)
       - A look-ahead penalty for the "regret" of not visiting a node now (approximated by how far it is from its nearest alternative)
    4. Use a weighted score: primarily nearest neighbor, but with a bias to keep the vehicle from going too far from the depot if capacity is running low.
    """
    # Convert to numpy arrays for fast operations
    unvisited = np.array(unvisited_nodes)
    if len(unvisited) == 0:
        return 0

    # Get demands for unvisited nodes
    node_demands = demands[unvisited]
    
    # Filter by capacity: only nodes whose demand fits in rest_capacity
    feasible_mask = node_demands <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0

    # Get distances from current node to feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Get distances from feasible nodes to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Calculate a "regret" score: how much worse it would be to visit this node later?
    # Approximation: if a node is far from the depot, and we have little capacity left, 
    # we might want to visit it soon before we're forced to go back to depot.
    # Simple heuristic: score = dist_from_current + alpha * dist_to_depot
    # Where alpha increases as rest_capacity decreases relative to total possible.
    
    # A more refined approach:
    # We want to minimize total route length. A good proxy is to choose the node that
    # minimizes (dist_from_current + dist_to_depot) but with a penalty for "stranding"
    # nodes that are far from the depot if we're running low on capacity.
    
    # Let's use a simple weighted sum:
    # score = dist_from_current + beta * dist_to_depot
    # beta = 0.5 * (1 - rest_capacity / max_capacity_estimate)
    # But we don't know max_capacity_estimate easily. Let's use a fixed beta that adapts.
    
    # Alternative: use the "nearest neighbor with depot bias"
    # If rest_capacity is low (say < 10), bias more towards nodes closer to depot.
    # If rest_capacity is high, use pure nearest neighbor.
    
    # Let's compute a dynamic beta based on how much capacity is left relative to the 
    # average demand of remaining nodes.
    if len(unvisited) > 0:
        avg_demand = np.mean(demands[unvisited])
    else:
        avg_demand = 1.0
    
    # How many more nodes can we roughly visit?
    remaining_capacity_ratio = rest_capacity / max(avg_demand, 1.0)
    
    # If we can only visit 1-2 more nodes, bias strongly towards depot
    # If we can visit many, bias weakly
    if remaining_capacity_ratio <= 2:
        beta = 0.8
    elif remaining_capacity_ratio <= 4:
        beta = 0.5
    elif remaining_capacity_ratio <= 8:
        beta = 0.3
    else:
        beta = 0.1

    # Compute scores
    scores = dist_from_current + beta * dist_to_depot
    
    # Choose the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
