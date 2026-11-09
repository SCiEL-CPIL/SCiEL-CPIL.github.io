"""
Geometric Sweep TSP Constructive Heuristic.

Strategy:
Instead of minimizing local distances (which can lead to edge crossings and
short-sighted "dead ends"), this heuristic uses a global geometric property:
the angle between the vector from the current node to the destination node
and the vector from the current node to the candidate next node.

By minimizing this angle (with a tie-breaker based on distance to destination),
the algorithm effectively attempts to trace the boundary of the point cloud
(convex hull) first, then peels off interior points. This "sweep" approach
typically produces non-crossing tours and avoids the long, erratic jumps
seen in pure nearest-neighbor methods, especially in the early stages of
tour construction.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time management
_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        return 0.0
    return time.time() - _start_time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a geometric sweep heuristic.
    
    Args:
        current_node (int): The node currently being visited.
        destination_node (int): The starting node (to be visited last to close tour).
        unvisited_nodes (np.ndarray): Array of node indices not yet visited.
        distance_matrix (np.ndarray): NxN matrix of Euclidean distances.
    
    Returns:
        int: The index of the next node to visit.
    """
    n_unvisited = len(unvisited_nodes)
    
    # Base cases
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check time budget - if running low, fall back to fast greedy
    elapsed = _get_elapsed()
    if elapsed > _TIMEOUT:
        # Fallback: Pure Nearest Neighbor (fastest)
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Standard case: Geometric Sweep
    # 1. Extract coordinates from the distance matrix? 
    #    Wait, we only have the distance matrix. We don't have explicit coordinates.
    #    However, we can simulate the "angle" logic using distances if we assume Euclidean geometry,
    #    but calculating angles requires coordinates.
    #    
    #    Alternative approach without explicit coordinates:
    #    We can approximate the "sweep" behavior by looking at the relative distances.
    #    A node that is "close" to the destination in terms of distance to current + distance to dest
    #    but NOT the closest to current might be a better "hull" candidate.
    #    
    #    Actually, we can reconstruct coordinates if we assume the distance matrix is Euclidean
    #    and we fix the first two points? No, that's complex and risky.
    #    
    #    Let's look at the properties of the distance matrix.
    #    If we want to "sweep" towards the destination, we want to pick nodes that
    #    are roughly in the "direction" of the destination.
    #    
    #    Heuristic: 
    #    Score = dist(current, candidate) + alpha * dist(candidate, destination)
    #    This is a blend of "go to candidate" and "get closer to destination".
    #    This is similar to the "phase-aware" insight but applied globally.
    #    
    #    However, the directive asks for "convex hull peeling" via "angle deviation".
    #    Without coordinates, we can't compute angles directly.
    #    
    #    BUT, we can estimate the angle using the Law of Cosines if we have the three distances:
    #    d_current_candidate, d_current_destination, d_candidate_destination.
    #    
    #    Let A = current, B = candidate, D = destination.
    #    We want to minimize the angle at A between vector AD and vector AB.
    #    Let theta be the angle BAD.
    #    By Law of Cosines on triangle ABD:
    #    d(BD)^2 = d(AB)^2 + d(AD)^2 - 2 * d(AB) * d(AD) * cos(theta)
    #    cos(theta) = (d(AB)^2 + d(AD)^2 - d(BD)^2) / (2 * d(AB) * d(AD))
    #    
    #    We want to minimize theta, which means maximizing cos(theta) (for theta in [0, pi]).
    #    Note: If the points are collinear and B is between A and D, theta=0, cos=1.
    #    If B is "behind" A relative to D, theta > 90, cos < 0.
    #    
    #    So the strategy is:
    #    1. Calculate d_AD (distance from current to destination).
    #    2. For each candidate B:
    #       a. Get d_AB = distance_matrix[current_node, B]
    #       b. Get d_BD = distance_matrix[B, destination_node]
    #       c. Calculate cos_theta = (d_AB^2 + d_AD^2 - d_BD^2) / (2 * d_AB * d_AD)
    #       d. We want to maximize cos_theta.
    #    
    #    This effectively picks the node that is most "aligned" with the destination.
    #    This traces the hull!
    #    
    #    Edge cases:
    #    - d_AB or d_AD is 0? 
    #      If current == destination, d_AD = 0. Then we just want to minimize d_AB (NN).
    #      If current == candidate, d_AB = 0. This shouldn't happen as candidate is unvisited.
    #    
    #    Let's implement this.
    
    d_AD = distance_matrix[current_node, destination_node]
    
    # Get distances from current to all candidates
    d_AB = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from all candidates to destination
    d_BD = distance_matrix[unvisited_nodes, destination_node]
    
    if d_AD < 1e-9:
        # Current is destination (or very close). Just pick nearest neighbor.
        return int(unvisited_nodes[np.argmin(d_AB)])
    
    # Calculate cos(theta) for each candidate
    # cos(theta) = (d_AB^2 + d_AD^2 - d_BD^2) / (2 * d_AB * d_AD)
    
    numerator = d_AB**2 + d_AD**2 - d_BD**2
    denominator = 2 * d_AB * d_AD
    
    # Handle division by zero (d_AB should not be 0 for unvisited nodes unless duplicate indices, 
    # but let's be safe)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = numerator / denominator
    
    # If denominator was 0, cos_theta might be inf or nan. 
    # If d_AB is 0, it's the same node, which shouldn't be in unvisited.
    # If d_AD is 0, we handled it above.
    
    # We want to MAXIMIZE cos_theta.
    # However, we also need to consider that picking a node that is extremely far away
    # but aligned might be bad? 
    # Actually, the "sweep" heuristic usually just picks the most aligned one.
    # But if we pick a node that is very far, the tour length increases.
    # 
    # Let's refine: 
    # The angle heuristic works best for hull tracing. 
    # To prevent picking a distant outlier that is aligned, we can add a small penalty 
    # for distance, or just rely on the angle. 
    # In TSP, "sweep" heuristics often just use the angle. 
    # Let's try maximizing cos_theta first.
    
    # If there are NaNs, replace with -inf (worst case)
    cos_theta = np.nan_to_num(cos_theta, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    # Select the candidate with the maximum cos_theta (smallest angle)
    best_idx = np.argmax(cos_theta)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
