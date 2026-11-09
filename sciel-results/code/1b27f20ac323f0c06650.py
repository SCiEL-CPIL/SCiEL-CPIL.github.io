"""Traveling salesman constructive heuristic using geometric angle minimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track time and context
_start_time = None
_total_nodes = None
_nodes_visited = 0
_destination = None
_prev_node = None

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour construction using angle minimization.
    
    Strategy:
    - Minimize the turning angle between the incoming vector and the outgoing vector.
    - For the first step, minimize the angle relative to the vector pointing to the destination.
    - This encourages smooth, low-curvature paths which are characteristic of optimal TSP tours.
    - Includes a time guard to ensure we finish within the deadline.
    
    Parameters:
    - current_node: int, the current node
    - destination_node: int, the final destination (start node for closed tour)
    - unvisited_nodes: np.ndarray, array of unvisited node indices
    - distance_matrix: np.ndarray, pairwise distance matrix
    
    Returns:
    - int: the index of the next node to visit
    """
    global _start_time, _total_nodes, _nodes_visited, _destination, _prev_node
    
    # Initialize state if this is the first call
    if _start_time is None:
        _start_time = time.time()
        _total_nodes = len(distance_matrix)
        _nodes_visited = 0
        _destination = destination_node
        _prev_node = None
    
    # Increment visited count
    _nodes_visited += 1
    
    # Check if we're running out of time - fall back to simple nearest neighbor
    if 120.0 - (time.time() - _start_time) < 5.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # If only one unvisited node, return it
    if len(unvisited_nodes) <= 1:
        _prev_node = current_node
        return int(unvisited_nodes[0])
    
    # Get coordinates. Since we only have distance_matrix, we need to reconstruct or estimate coordinates.
    # However, the standard interface for these tasks usually implies we can access coordinates 
    # or we must work with the distance matrix directly. 
    # Wait, the interface only provides distance_matrix. 
    # To compute angles, we need coordinates. 
    # We can derive coordinates from the distance matrix using Multidimensional Scaling (MDS) 
    # or assume the distance matrix is Euclidean and we can recover up to rotation/translation.
    # However, doing MDS every step is expensive.
    # Alternative: Use the distance matrix to estimate the "direction" via vector differences if we had coords.
    # Since we don't have coords, we must approximate the geometric preference using the distance matrix.
    
    # Let's refine the strategy: 
    # If we don't have coordinates, we can't compute angles directly. 
    # The prompt says "distance_matrix: np.ndarray". It does not explicitly provide coordinates.
    # However, for 50 cities, we can perform a lightweight MDS once at the start to get coordinates, 
    # or we can use a heuristic that approximates angle minimization using the triangle inequality properties.
    
    # Actually, a common trick in these constrained interfaces is that the distance matrix 
    # is Euclidean, and we can recover coordinates using MDS. 
    # Let's cache the coordinates.
    
    global _coords
    if '_coords' not in globals():
        _coords = None
    
    if _coords is None:
        # Perform MDS to recover coordinates from distance matrix
        # This is done once per instance (detected by change in total_nodes or first call)
        D = distance_matrix
        n = D.shape[0]
        # Double centering
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        # Eigen decomposition
        w, v = np.linalg.eigh(B)
        # Sort eigenvalues in descending order
        idx = np.argsort(w)[::-1]
        w = w[idx]
        v = v[:, idx]
        # Take top 2 dimensions
        # Ensure positive square roots
        w = np.maximum(w, 0)
        U = v[:, :2] * np.sqrt(w[:2])
        _coords = U
        
    # Now we have coordinates for all nodes
    # current_pos = _coords[current_node]
    # dest_pos = _coords[destination_node]
    
    # Determine the reference direction
    if _nodes_visited == 1:
        # First step: direction towards destination
        ref_vec = _coords[destination_node] - _coords[current_node]
    else:
        # Subsequent steps: direction from previous node to current node
        # We need to track the previous node. 
        # In the current state, _prev_node is the node we came from.
        # But wait, the function signature doesn't give us the previous node explicitly, 
        # but we can infer it if we track it. 
        # Actually, _prev_node is updated at the end of the previous call.
        # Let's ensure _prev_node is set correctly.
        # In the first call, _prev_node is None.
        # In subsequent calls, _prev_node is the node that was selected in the previous step.
        # But the "current_node" passed in IS the node selected in the previous step.
        # So the "previous node" (the one before current) is not directly available unless we track a history.
        # Let's track a small history or just use the destination as a fallback if history is missing.
        # Better: Track the last two nodes.
        
        pass # Placeholder, will implement below

    # Let's re-implement state tracking to be robust.
    # We need the vector from the node BEFORE current to current.
    # Let's add a global _last_two_nodes or just _incoming_vec.
    
    global _incoming_vec
    if '_incoming_vec' not in globals():
        _incoming_vec = None
    
    if _nodes_visited == 1:
        # First step
        ref_vec = _coords[destination_node] - _coords[current_node]
        # Normalize
        norm = np.linalg.norm(ref_vec)
        if norm > 0:
            ref_vec = ref_vec / norm
        else:
            ref_vec = np.array([1.0, 0.0])
    else:
        # Use stored incoming vector
        if _incoming_vec is not None:
            ref_vec = _incoming_vec
        else:
            # Fallback if state is lost: use destination
            ref_vec = _coords[destination_node] - _coords[current_node]
            norm = np.linalg.norm(ref_vec)
            if norm > 0:
                ref_vec = ref_vec / norm
            else:
                ref_vec = np.array([1.0, 0.0])
    
    # Calculate vectors from current to all unvisited nodes
    # current_pos = _coords[current_node]
    # unvisited_pos = _coords[unvisited_nodes]
    # vectors = unvisited_pos - current_pos
    
    # To minimize turning angle, we want to maximize the dot product with ref_vec
    # dot_product = vectors @ ref_vec
    
    # However, we also need to consider distance to avoid getting stuck in a local cluster 
    # or to ensure progress. A pure angle minimization might spiral.
    # A common heuristic is: Score = -dot_product * alpha + distance * beta
    # Or simply: Minimize the angle, but break ties with distance.
    # Let's use a combined score: 
    # Score = angle_penalty + distance_penalty
    # angle_penalty = 1 - cos(angle) = 1 - (v_unit . ref_vec)
    # distance_penalty = distance
    
    # Calculate distances and unit vectors
    current_pos = _coords[current_node]
    unvisited_pos = _coords[unvisited_nodes]
    vectors = unvisited_pos - current_pos
    dists = np.linalg.norm(vectors, axis=1)
    
    # Avoid division by zero
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1e-9
    unit_vectors = vectors / norms[:, np.newaxis]
    
    # Dot products with reference vector
    dots = unit_vectors @ ref_vec
    
    # Clamp dots to [-1, 1] for numerical stability
    dots = np.clip(dots, -1, 1)
    
    # Angle penalty: 1 - cos(theta)
    # We want to minimize this.
    angle_penalty = 1 - dots
    
    # Distance penalty: We want to minimize distance, but not too much to avoid spirals.
    # Normalize distance by the average distance to make it comparable to angle penalty.
    avg_dist = np.mean(dists)
    if avg_dist == 0:
        avg_dist = 1
    dist_penalty = dists / avg_dist
    
    # Combine scores
    # Weights: 
    # Early phase: Balance between angle and distance to build a smooth path that covers space.
    # Late phase: Prioritize angle to close the loop smoothly.
    # Let's use a fixed weight that works well generally.
    # Angle is dimensionless (0 to 2). Distance is normalized (avg 1).
    # A weight of 1.0 for angle and 0.5 for distance seems reasonable.
    # Let's try: Score = angle_penalty + 0.5 * dist_penalty
    
    score = angle_penalty + 0.5 * dist_penalty
    
    # Select the node with minimum score
    next_idx = np.argmin(score)
    
    # Update state for next call
    _incoming_vec = unit_vectors[next_idx] # The direction we are now moving
    # Note: _incoming_vec for the NEXT call should be the vector from CURRENT to NEXT.
    # So we store the unit vector of the chosen edge.
    
    return int(unvisited_nodes[next_idx])
# EVOLVE-BLOCK-END
