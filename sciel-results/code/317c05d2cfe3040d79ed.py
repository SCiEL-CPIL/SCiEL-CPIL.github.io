"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds
# We want to return before the hard limit; use 115s as safety margin
_SOFT_DEADLINE = 115.0


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour.
    
    Uses a phase-aware heuristic:
    - Early phase: mostly nearest neighbor
    - Late phase: blend in destination proximity to avoid long closing edge
    
    Parameters:
    - current_node: int, the current city
    - destination_node: int, the city we need to return to (start of tour)
    - unvisited_nodes: np.ndarray of unvisited city indices
    - distance_matrix: np.ndarray of pairwise distances
    
    Returns:
    - int: the next city to visit
    """
    # Deadline check: if we're running out of time, fall back to fast nearest neighbor
    if time.time() > _SOFT_DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen per contract, but guard
        return current_node
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from destination to all unvisited (for closing edge consideration)
    dist_to_destination = distance_matrix[destination_node, unvisited_nodes]
    
    # Phase-aware weighting:
    # When few nodes remain, we care more about which node to pick to minimize the 
    # final edge back to destination. When many remain, pure NN is good.
    
    # Total nodes in the tour is 50 (from problem description).
    # We can estimate the phase by how many are left.
    # Let's assume total is around 50; we can infer from the size of the full matrix.
    total_nodes = distance_matrix.shape[0]
    
    # Fraction of tour remaining
    fraction_remaining = n_unvisited / max(total_nodes, 1)
    
    # Weight for destination proximity: increases as we approach the end
    # For the last few picks, we want to strongly consider the closing edge.
    # Use a smooth ramp: weight goes from 0 to ~0.5 over the last 20% of the tour.
    
    if fraction_remaining > 0.3:
        # Early/mid phase: mostly nearest neighbor
        dest_weight = 0.0
    else:
        # Late phase: blend in destination proximity
        # fraction_remaining is in [0, 0.3]
        # Map to weight [0, 0.5]
        dest_weight = 0.5 * (1.0 - fraction_remaining / 0.3)
    
    # Combined score: lower is better
    # score = dist_from_current + dest_weight * dist_to_destination
    scores = dist_from_current + dest_weight * dist_to_destination
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
