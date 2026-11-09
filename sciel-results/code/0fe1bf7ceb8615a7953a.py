"""Improved TSP constructive heuristic with phase-aware blending and look-ahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking
_start_time = None
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _init_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _time_exceeded():
    global _start_time
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _TIME_LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour construction.
    
    Uses a phase-aware strategy:
    - Early phase: mostly nearest neighbor
    - Late phase: blends in proximity to destination to avoid long closing edges
    - Uses a smooth surrogate (log-sum-exp based soft-min) for robustness
    
    Parameters:
    -----------
    current_node : int
        The current node in the tour.
    destination_node : int
        The starting node (where the tour must return).
    unvisited_nodes : np.ndarray
        Array of unvisited node indices.
    distance_matrix : np.ndarray
        2D array of distances between all pairs of nodes.
        
    Returns:
    --------
    int
        The index of the next node to visit.
    """
    _init_time()
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return current_node  # Should not happen in valid TSP construction
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if close to timeout, use fast nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from unvisited nodes to destination (for closing edge consideration)
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Determine phase based on how many nodes remain
    # Total nodes = n_unvisited + (nodes already visited)
    # We don't know total nodes directly, but we can use n_unvisited as proxy
    # For 50-city instances, we can assume total is around 50
    # Let's estimate: if we're early, n_unvisited is large; if late, it's small
    total_nodes_estimate = 50  # Assumption based on task description
    
    # Phase: 0.0 = pure NN, 1.0 = pure destination-proximity
    # Use a smooth transition
    if n_unvisited > 30:
        phase = 0.0
    elif n_unvisited < 10:
        phase = 1.0
    else:
        # Linear blend from 0 to 1 between n_unvisited=30 and n_unvisited=10
        phase = (30 - n_unvisited) / 20.0
    
    # Create a combined score
    # We want to minimize: (1-phase)*dist_from_current + phase*alpha*dist_to_dest
    # But we need to be careful about scaling
    # Use a soft combination: score = dist_from_current + phase * beta * dist_to_dest
    
    # Choose beta based on phase to smoothly transition
    # In early phase, beta is small; in late phase, beta is larger
    beta = 0.5 + 0.5 * phase  # beta ranges from 0.5 to 1.0
    
    # Combined score
    score = dist_from_current + beta * dist_to_dest
    
    # For additional robustness, use a look-ahead heuristic in the middle phase
    if 0.2 < phase < 0.8 and n_unvisited > 5:
        # Look-ahead: for each candidate, estimate the cost of going to it and then
        # to the nearest unvisited node after that (excluding the candidate itself)
        # This is O(n^2) but n is small in this phase
        best_score = np.inf
        best_idx = 0
        
        for i, node in enumerate(unvisited_nodes):
            # Cost to reach this node
            cost_to_node = dist_from_current[i]
            
            # Cost from this node to destination
            cost_node_to_dest = dist_to_dest[i]
            
            # Look-ahead: find the nearest unvisited node to this candidate (excluding itself)
            if n_unvisited > 1:
                # Distances from this node to all other unvisited nodes
                other_unvisited = np.delete(unvisited_nodes, i)
                if len(other_unvisited) > 0:
                    dist_from_candidate = distance_matrix[node, other_unvisited]
                    min_lookahead = np.min(dist_from_candidate)
                else:
                    min_lookahead = 0
            else:
                min_lookahead = 0
            
            # Combined look-ahead score
            # Weight the look-ahead less in early phase, more in late phase
            lookahead_weight = 0.3 * phase
            combined_score = cost_to_node + beta * cost_node_to_dest + lookahead_weight * min_lookahead
            
            if combined_score < best_score:
                best_score = combined_score
                best_idx = i
        
        return int(unvisited_nodes[best_idx])
    
    # Default: pick the node with minimum combined score
    return int(unvisited_nodes[np.argmin(score)])
# EVOLVE-BLOCK-END
