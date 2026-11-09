import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill

# Cache for coordinates derived from distance matrix (Euclidean assumption)
_COORD_CACHE = {}


def _get_coordinates(distance_matrix):
    """
    Recover 2D coordinates from a Euclidean distance matrix using MDS.
    Returns (coords, success).
    """
    key = id(distance_matrix)
    if key in _COORD_CACHE:
        return _COORD_CACHE[key]
    
    n = distance_matrix.shape[0]
    try:
        # Use classical MDS to recover coordinates
        D2 = distance_matrix ** 2
        # Centering matrix
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Take top 2 eigenvalues
        idx = np.argsort(eigenvalues)[::-1][:2]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Coordinates
        coords = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        _COORD_CACHE[key] = (coords, True)
        return coords, True
    except Exception:
        _COORD_CACHE[key] = (None, False)
        return None, False


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Uses a phase-aware strategy:
    1. For the majority of the tour (many unvisited nodes), use pure Nearest Neighbor.
       This avoids the pitfalls of complex heuristics that might distort early tour construction.
    2. For the final few nodes, use a destination-aware heuristic to minimize the closing edge.
    """
    # Time guard - if we're running out of time, fall back to greedy
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Threshold for switching to destination-aware mode
    # Based on findings: pure NN is good for the first ~40 picks (out of 50).
    # So we switch when <= 10 nodes remain.
    THRESHOLD = 10
    
    if n_unvisited > THRESHOLD:
        # Phase 1: Pure Nearest Neighbor
        # This is the structural change: stop using complex adaptive weights early on.
        d_curr = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(d_curr)
        return int(unvisited_nodes[best_idx])
    
    # Phase 2: Destination-aware for the final stretch
    # We want to minimize the total cost of the remaining path + closing edge.
    # Heuristic: score = d(current, cand) + lambda * d(cand, destination)
    # As n_unvisited gets smaller, we care more about the destination to close the loop.
    
    if n_unvisited == 2:
        # Exact check for 2 nodes
        u, v = unvisited_nodes
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
        
    if n_unvisited == 3:
        # Exact check for 3 nodes (6 permutations)
        u, v, w = unvisited_nodes
        perms = [
            (u, v, w), (u, w, v),
            (v, u, w), (v, w, u),
            (w, u, v), (w, v, u)
        ]
        best_cost = float('inf')
        best_first = u
        for p in perms:
            c = (distance_matrix[current_node, p[0]] + 
                 distance_matrix[p[0], p[1]] + 
                 distance_matrix[p[1], p[2]] + 
                 distance_matrix[p[2], destination_node])
            if c < best_cost:
                best_cost = c
                best_first = p[0]
        return int(best_first)
        
    # For 4 to THRESHOLD nodes:
    # Use a weighted combination of distance to current and distance to destination.
    # The weight on destination should increase as we get closer to finishing.
    # When n_unvisited = 10, we are still a bit out, so weight is moderate.
    # When n_unvisited = 4, we are close, so weight is high.
    
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive lambda: 
    # At n_unvisited=10, lambda ~ 0.5
    # At n_unvisited=4, lambda ~ 1.5
    # Linear interpolation
    # progress = 1 - (n_unvisited - 4) / (10 - 4)
    # lambda = 0.5 + 1.0 * progress
    # Alternatively, just use a fixed weight that works well for the "tail"
    # Let's try a weight that increases as n_unvisited decreases.
    # lambda = 2.0 - 0.1 * n_unvisited? 
    # If n=10, lambda=1.0. If n=4, lambda=1.6.
    
    # Let's use a simpler heuristic:
    # Minimize d(current, cand) + d(cand, dest) is not quite right because we have to visit others.
    # But for the last few, it's a good proxy for "don't end up far from destination".
    
    # A better structural approach for the tail:
    # Estimate the cost of the remaining tour if we pick cand.
    # Roughly: d(current, cand) + (sum of nearest neighbor edges for remaining) + d(last, dest).
    # This is expensive.
    
    # Let's stick to the weighted sum which is fast and effective for small N.
    # We want to bias towards nodes that are close to the destination if they are not too far from current.
    
    # Let's try: score = d_curr + alpha * d_dest
    # Where alpha increases as n_unvisited decreases.
    # alpha = 1.0 when n_unvisited == 10
    # alpha = 2.0 when n_unvisited == 4
    
    # Linear map:
    # n=10 -> alpha=1.0
    # n=4 -> alpha=2.0
    # slope = (2.0 - 1.0) / (4 - 10) = 1 / -6 = -0.166
    # alpha = 1.0 + (-0.166) * (n_unvisited - 10)
    # alpha = 1.0 + 0.166 * (10 - n_unvisited)
    
    alpha = 1.0 + 0.166 * (10 - n_unvisited)
    
    scores = d_curr + alpha * d_dest
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
