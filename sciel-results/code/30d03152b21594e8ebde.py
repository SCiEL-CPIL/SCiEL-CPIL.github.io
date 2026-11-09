"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a geometric convex-hull-peeling heuristic.
    
    Strategy:
    1. Calculate the "excess" distance for each candidate: 
       excess[i] = dist(current, i) + dist(i, destination) - dist(current, destination).
       This measures the deviation from the straight line current->destination.
       Minimizing this keeps the tour "straight" towards the destination.
    
    2. Calculate "hull-ness" for each candidate:
       Use the eccentricity (max distance to other unvisited nodes) as a proxy
       for being on the convex hull. Nodes on the hull have higher eccentricity.
    
    3. Combine scores:
       score[i] = excess[i] - lambda * hull_norm[i] * scale
       We prefer nodes with low excess (straight) and high hull-ness (outer).
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(0)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Distances from current and destination to all unvisited nodes
    dist_current = distance_matrix[current_node, unvisited_nodes]
    dist_dest = distance_matrix[destination_node, unvisited_nodes]
    dist_curr_dest = distance_matrix[current_node, destination_node]
    
    # Excess distance: penalty for deviating from the straight line C->D
    # If i is on the segment CD, excess is 0.
    excess = dist_current + dist_dest - dist_curr_dest
    
    # Ensure excess is non-negative (numerical stability)
    excess = np.maximum(excess, 0.0)
    
    # Calculate hull-ness using eccentricity (max distance to other unvisited)
    # This identifies "extreme" points which are likely on the convex hull
    unvisited_dist_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # Mask diagonal to avoid self-distance (which is 0) affecting max if we used min, 
    # but for max, self-distance 0 is fine as long as there are other nodes.
    # However, strictly speaking, eccentricity is max distance to *other* nodes.
    # Since distance_matrix diagonal is 0, and n_unvisited > 1, max will be > 0.
    eccentricity = np.max(unvisited_dist_matrix, axis=1)
    
    # Normalize eccentricity to [0, 1] to make it comparable with excess (scaled)
    ecc_min = np.min(eccentricity)
    ecc_max = np.max(eccentricity)
    if ecc_max > ecc_min:
        hull_norm = (eccentricity - ecc_min) / (ecc_max - ecc_min)
    else:
        hull_norm = np.zeros(n_unvisited)
    
    # Scale factor: use the distance between current and destination as a reference scale
    # This makes the hull bonus proportional to the step size
    scale = dist_curr_dest if dist_curr_dest > 0 else 1.0
    
    # Lambda: weight for hull-ness
    # We want to strongly prefer hull nodes, but not ignore the straight-line constraint.
    # A value around 1.0-2.0 seems appropriate.
    lambda_hull = 1.5
    
    # Score: lower is better
    # We minimize excess, and subtract the hull bonus.
    score = excess - lambda_hull * hull_norm * scale
    
    # Select the node with minimum score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
