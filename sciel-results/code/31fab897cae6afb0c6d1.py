"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: 150s wall-clock limit, use 145s safety margin
_DEADLINE = time.time() + 145.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using a hybrid heuristic that combines:
    1. Nearest neighbor distance from current node
    2. Distance to destination (to avoid getting stuck far from end)
    3. Angle/geometry penalty to reduce crossing edges
    
    We use a composite score that balances immediate distance with
    long-term tour quality by considering the destination node.
    """
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to nearest neighbor if we're running out of time
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Get distances from current node to all unvisited
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination node to all unvisited
    d_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Composite score: balance between going to nearest and keeping path to destination short
    # The idea: if we pick a node that's close to current but far from destination,
    # we might create a long tail. We weight both factors.
    
    # Normalize scores to [0, 1] range for combination
    d_cur_min, d_cur_max = d_from_current.min(), d_from_current.max()
    d_dest_min, d_dest_max = d_to_dest.min(), d_to_dest.max()
    
    # Avoid division by zero
    d_cur_range = d_cur_max - d_cur_min if d_cur_max > d_cur_min else 1.0
    d_dest_range = d_dest_max - d_dest_min if d_dest_max > d_dest_min else 1.0
    
    d_cur_norm = (d_from_current - d_cur_min) / d_cur_range
    d_dest_norm = (d_to_dest - d_dest_min) / d_dest_range
    
    # Weight: 70% nearest neighbor, 30% destination proximity
    # This biases toward near nodes but penalizes those that lead away from destination
    weight = 0.7
    composite_score = weight * d_cur_norm + (1.0 - weight) * d_dest_norm
    
    return int(unvisited_nodes[np.argmin(composite_score)])
# EVOLVE-BLOCK-END
