"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_SECONDS = 145.0  # safety margin before 150s kill


def _check_deadline():
    """Return True if we should stop and fall back to simple heuristic."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > _DEADLINE_SECONDS


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a connectivity-aware heuristic.
    
    For each candidate, we estimate the cost of adding it to the partial tour
    and the potential cost of closing the tour. We use a combination of:
    1. Distance from current node to candidate
    2. Distance from candidate to the farthest unvisited node (connectivity)
    3. Distance from candidate to destination (for final closure)
    
    This helps avoid creating disconnected subtours and ensures the partial
    tour remains a single path that can be closed efficiently.
    """
    # If no unvisited nodes, return -1 or handle appropriately
    if len(unvisited_nodes) == 0:
        return -1
    
    # Check deadline - if approaching, use simple nearest neighbor
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    # If only one node left, return it
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Precompute distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Precompute distances from destination to all unvisited
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For each candidate, compute a connectivity-aware score
    # We want to minimize: dist(current, candidate) + alpha * connectivity_penalty
    # where connectivity_penalty encourages keeping the tour connected
    
    # Strategy: For each candidate i, estimate the cost of completing the tour
    # by looking at the minimum spanning tree of remaining nodes plus connections
    
    # A simpler but effective heuristic:
    # Score = dist(current, candidate) + beta * min_dist(candidate, other_unvisited)
    # This encourages choosing nodes that are close to others (connected)
    
    # But even better: use a 2-step lookahead with connectivity
    # For each candidate c:
    #   cost = dist(current, c) + min_{c' in unvisited\{c}} [dist(c, c') + dist(c', destination)]
    # This is O(n^2) per call, but n is at most 50, so 2500 ops is fine
    
    if n_unvisited > 2:
        # 2-step lookahead with connectivity awareness
        scores = np.zeros(n_unvisited)
        
        # Precompute distance matrix among unvisited nodes
        dist_among = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        
        for i in range(n_unvisited):
            # Direct cost
            direct_cost = dist_from_current[i]
            
            # Lookahead: best next node after choosing i
            if n_unvisited > 2:
                # Cost of going from i to some j, then j to destination
                # We want min over j != i of dist(i,j) + dist(j, destination)
                dist_from_i = dist_among[i, :]
                dist_from_i[i] = np.inf  # exclude self
                
                # Combine with distance to destination
                combined = dist_from_i + dist_to_dest
                lookahead_cost = np.min(combined)
            else:
                lookahead_cost = dist_to_dest[i]
            
            # Weighted combination
            # Higher weight on lookahead to avoid bad choices
            alpha = 0.7  # weight on lookahead
            scores[i] = (1 - alpha) * direct_cost + alpha * lookahead_cost
        
        best_idx = np.argmin(scores)
    else:
        # For 2 nodes, just pick the one that leads to shorter total tour
        # Option 1: go to node 0, then node 1, then destination
        # Option 2: go to node 1, then node 0, then destination
        cost_option1 = dist_from_current[0] + distance_matrix[unvisited_nodes[0], unvisited_nodes[1]] + dist_to_dest[1]
        cost_option2 = dist_from_current[1] + distance_matrix[unvisited_nodes[1], unvisited_nodes[0]] + dist_to_dest[0]
        
        if cost_option1 <= cost_option2:
            best_idx = 0
        else:
            best_idx = 1
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
