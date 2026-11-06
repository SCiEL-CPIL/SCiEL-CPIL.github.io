"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using 2-step lookahead with adaptive weights:
    score = d(current, candidate) + alpha * min_d(candidate->other_unvisited->back) 
            + beta * d(candidate, destination)
    
    Uses a stronger lookahead that considers the cost of returning from the 
    next node, plus a closure penalty to avoid stranding.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, fall back to pure nearest neighbor
    if time.time() - _start_time > TIME_LIMIT:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Immediate edge costs from current to each candidate
    immediate_costs = distance_matrix[current_node, unvisited_nodes]
    
    # 2-step lookahead: for each candidate, find the cheapest next hop AND the cost 
    # from that next hop back to the destination (or to another unvisited node)
    # This is more expensive but gives better global awareness
    
    # Create submatrix of distances between unvisited nodes
    dist_sub = distance_matrix[unvisited_nodes][:, unvisited_nodes]
    
    # For each candidate i, compute min over j != i of: d(i,j) + d(j, dest)
    # This represents the cost of going from candidate to some next node and then to destination
    # We add this as a lookahead penalty
    
    # dist_to_dest: cost from each unvisited node to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For 2-step lookahead: cost(i) = min_{j != i} [d(i,j) + d(j, dest)]
    # Vectorized: add dist_to_dest to each column of dist_sub
    # lookahead_cost[i] = min_j (dist_sub[i,j] + dist_to_dest[j]) for j != i
    lookahead_matrix = dist_sub + dist_to_dest[np.newaxis, :]
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(lookahead_matrix, np.inf)
    two_step_lookahead = np.min(lookahead_matrix, axis=1)
    
    # Also compute 1-step min next hop for a simpler penalty
    min_next_hop = np.min(dist_sub, axis=1) if n_unvisited > 1 else np.zeros(n_unvisited)
    if n_unvisited > 1:
        # Re-set diagonal properly for min_next_hop
        dist_sub_copy = dist_sub.copy()
        np.fill_diagonal(dist_sub_copy, np.inf)
        min_next_hop = np.min(dist_sub_copy, axis=1)
    
    # Closure penalty: cost from candidate to destination
    dest_costs = dist_to_dest
    
    # Adaptive weights: stronger lookahead when fewer nodes remain (tour is closing)
    # Early on, prioritize immediate cost; later, prioritize closure
    progress = (50 - n_unvisited) / 50.0  # 0 at start, 1 at end
    alpha = 0.3 + 0.4 * progress  # 0.3 to 0.7
    beta = 0.2 + 0.3 * progress   # 0.2 to 0.5
    
    # Combine: immediate + alpha * two_step_lookahead + beta * dest_costs
    # Scale the lookahead to avoid double-counting
    scores = immediate_costs + alpha * two_step_lookahead + beta * dest_costs
    
    # Pick the candidate with minimum combined score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
