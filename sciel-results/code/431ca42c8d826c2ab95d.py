"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a geometric convex-hull-peeling heuristic.
    
    The idea: instead of picking the nearest unvisited node, we try to follow
    the boundary of the unvisited point cloud. We compute the direction from
    current_node toward destination_node (or the centroid of unvisited nodes as
    a fallback), and then select the unvisited node that best continues this
    direction while staying close to the convex hull of the unvisited set.
    
    Specifically, we:
    1. Compute the vector from current_node to the centroid of unvisited nodes
       (as a proxy for "where we're going" in the tour).
    2. For each unvisited node, compute the angle between the current direction
       and the vector from current_node to that node.
    3. Also compute the distance from current_node to each unvisited node.
    4. Score each candidate by a combination of:
       - Angle deviation (prefer nodes that continue the current direction)
       - Distance (prefer nodes that are not too far, but not necessarily the closest)
       - A "hull-ness" factor: nodes on the convex hull of the unvisited set are
         preferred early in the tour.
    
    This encourages tracing the convex hull before filling interior points.
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        # No unvisited nodes; return destination or current (shouldn't happen in valid TSP)
        return int(0)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get coordinates if available, otherwise work purely with distance matrix
    # Since we only have distance_matrix, we need to infer geometry or use
    # distance-based proxies for angle and hull-ness.
    
    # Direction vector proxy: use the vector from current to destination
    # But we don't have coordinates, only distances. So we need a different approach.
    
    # Alternative approach using only distance matrix:
    # 1. Compute distance from current to each unvisited node
    # 2. Compute distance from destination to each unvisited node  
    # 3. Use the "angle" proxy: nodes that are roughly equidistant from current
    #    and destination (or where the path current->node->destination is nearly
    #    straight) are preferred.
    
    # Actually, let's think differently. With only distance matrix, we can:
    # - Compute dist_current[i] = distance from current to unvisited[i]
    # - Compute dist_dest[i] = distance from destination to unvisited[i]
    # - Compute dist_curr_dest = distance from current to destination
    
    # A node that lies "between" current and destination on a straight line would
    # satisfy: dist_current[i] + dist_dest[i] ≈ dist_curr_dest
    # Nodes that deviate from this line have larger dist_current[i] + dist_dest[i]
    
    # For convex hull peeling, we want to follow the boundary. A good proxy:
    # Among unvisited nodes, prefer those that are "extreme" in some sense.
    
    # Let's use a combined score:
    # score[i] = alpha * dist_current[i] + beta * (dist_current[i] + dist_dest[i] - dist_curr_dest)
    # The second term penalizes nodes that are not on the "line" between current and dest.
    
    # But for convex hull tracing, we might want to prefer nodes that are FAR from
    # the centroid of unvisited nodes (i.e., on the hull). We can approximate this
    # by looking at how far each node is from all other unvisited nodes.
    
    # Simpler approach: use a "turning angle" heuristic.
    # We want to continue in the direction we're heading. Since we don't have
    # explicit coordinates, let's use the previous direction if available.
    
    # Actually, the function signature doesn't give us previous direction.
    # Let's use destination as the target direction.
    
    dist_current = distance_matrix[current_node, unvisited_nodes]
    dist_dest = distance_matrix[destination_node, unvisited_nodes]
    dist_curr_dest = distance_matrix[current_node, destination_node]
    
    # Heuristic: Minimize "backtracking" relative to the destination (start node).
    # We prefer nodes that continue the direction from destination to current.
    # This corresponds to maximizing the angle at current_node in triangle (dest, current, next)
    # such that the path dest->current->next is as straight as possible.
    # Straightness proxy: (dist_dest_to_next) should be approx (dist_dest_to_current + dist_current_to_next).
    # So we minimize the slack: dist_curr_dest + dist_current - dist_dest
    # If next is "beyond" current, slack is ~0.
    # If next is "behind" current (towards dest), slack is negative (bad).
    # So we want to MINIMIZE (dist_curr_dest + dist_current - dist_dest).
    
    straightness_penalty = dist_curr_dest + dist_current - dist_dest
    
    # "Hull-ness": nodes that are far from the centroid of unvisited nodes
    if n_unvisited > 1:
        unvisited_dist_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        avg_dist_to_others = unvisited_dist_matrix.sum(axis=1) / (n_unvisited - 1)
        hull_factor = avg_dist_to_others
    else:
        hull_factor = np.zeros(n_unvisited)
        
    # Normalize hull_factor to be comparable to straightness_penalty
    # straightness_penalty is in units of distance.
    # hull_factor is in units of distance.
    
    # Score: lower is better.
    # We want low straightness_penalty (straight path) and high hull_factor (extreme point).
    # score = straightness_penalty - lambda * hull_factor
    
    # Tune lambda to balance straightness vs hull-ness
    # If lambda is too high, we might pick a very hull-like node that causes a sharp turn.
    # If lambda is too low, we might pick a node that is straight but interior.
    lambda_hull = 0.5
    
    score = straightness_penalty - lambda_hull * hull_factor
    
    # Select the node with minimum score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
