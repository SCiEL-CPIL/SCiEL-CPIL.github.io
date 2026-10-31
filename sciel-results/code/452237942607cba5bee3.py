"""Geometric Convex Hull Peeling Strategy for TSP Constructive Heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for caching coordinates and deadline management
_COORD_CACHE = {}
_DEADLINE = None
_START_TIME = None

def _get_instance_id(distance_matrix):
    """Create a lightweight fingerprint for the distance matrix."""
    n = distance_matrix.shape[0]
    if n > 10:
        # Sample rows to create a hashable byte string
        sample = distance_matrix[::max(1, n//10), :].tobytes()
    else:
        sample = distance_matrix.tobytes()
    return hash(sample)

def _reconstruct_coords(distance_matrix, instance_id):
    """
    Reconstruct 2D coordinates from the distance matrix.
    Uses Multidimensional Scaling (MDS) via eigendecomposition of the 
    double-centered squared distance matrix.
    """
    if instance_id in _COORD_CACHE:
        return _COORD_CACHE[instance_id]
    
    n = distance_matrix.shape[0]
    # Use squared distances for MDS
    D2 = distance_matrix ** 2
    
    # Double centering: J = I - (1/n) * 11^T
    J = np.eye(n) - np.ones((n, n)) / n
    
    # B = -1/2 * J * D2 * J
    B = -0.5 * (J @ D2 @ J)
    
    # Eigendecomposition
    # We only need the top 2 eigenvalues/vectors
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 dimensions
    # Handle negative eigenvalues (numerical noise)
    eigenvalues = np.clip(eigenvalues, 0, None)
    
    U = eigenvectors[:, :2]
    S = np.sqrt(eigenvalues[:2])
    
    coords = U * S
    
    # Center the coordinates
    coords = coords - np.mean(coords, axis=0)
    
    _COORD_CACHE[instance_id] = coords
    return coords

def _select_next_node_geometric(current_node, destination_node, unvisited_nodes, distance_matrix, coords):
    """
    Select the next node using geometric convex hull peeling logic.
    
    Score = dist(current, candidate) + alpha * area(current, destination, candidate)
    """
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
        
    if len(unvisited_nodes) == 2:
        # For 2 nodes, just pick the one that results in a shorter total path 
        # considering the return to destination (or just nearest if destination is irrelevant/far)
        # The prompt says "destination_node" is passed. In TSP constructive, 
        # usually we don't have a fixed destination until the end, but here it's an argument.
        # If it's the final step, we want to minimize the closing edge.
        # However, the signature implies a general step.
        # Let's stick to the geometric formula or a simple heuristic for small N.
        # Formula: d(c, u) + alpha * area(c, dest, u)
        c = coords[current_node]
        dest = coords[destination_node]
        
        scores = []
        for u in unvisited_nodes:
            d = distance_matrix[current_node, u]
            # Area of triangle (c, dest, u)
            # 0.5 * |cross(dest - c, u - c)|
            v1 = dest - c
            v2 = coords[u] - c
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            area = 0.5 * abs(cross)
            
            # Alpha scaling: normalize area by typical distance scale?
            # For simplicity, use a fixed alpha relative to the mean distance.
            # Or just use the raw area if coordinates are in [0,1] or similar.
            # Let's estimate a scale factor.
            # Actually, let's just use a heuristic alpha. 
            # If coordinates are ~1.0 scale, area is ~0.1-1.0. Dist is ~0.1-1.5.
            # Let's try alpha = 1.0 first.
            score = d + 1.0 * area
            scores.append(score)
            
        return int(unvisited_nodes[np.argmin(scores)])

    c = coords[current_node]
    dest = coords[destination_node]
    
    # Vectorized calculation for all unvisited nodes
    u_coords = coords[unvisited_nodes]
    
    # Distances from current to unvisited
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate areas
    # v1 = dest - c
    v1 = dest - c
    # v2 = u - c for each u
    v2 = u_coords - c
    
    # Cross product z-component: v1_x * v2_y - v1_y * v2_x
    cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Scale the area term. 
    # To make the penalty meaningful, we should normalize it.
    # Let's use the mean distance from current to unvisited as a scale.
    mean_dist = np.mean(dists)
    if mean_dist > 0:
        normalized_areas = areas / mean_dist
    else:
        normalized_areas = areas
        
    # Heuristic: alpha controls the trade-off between "go close" and "keep perimeter tight"
    # A higher alpha penalizes "cutting inwards" more, encouraging hull-following.
    # For uniform random points, convex hull peeling is a strong heuristic.
    alpha = 1.5
    
    scores = dists + alpha * normalized_areas
    
    # Find the unvisited node with the minimum score
    min_idx = np.argmin(scores)
    return int(unvisited_nodes[min_idx])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE, _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill
        
    # Deadline guard: fall back to nearest neighbor if time is up
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        # Special case for last 2 nodes: ensure we don't create a crossing if possible, 
        # or just minimize the immediate step + closing step.
        # Since we don't know the full tour context easily without history, 
        # we use a simple 2-node lookahead.
        u0, u1 = unvisited_nodes[0], unvisited_nodes[1]
        # Cost if we pick u0: d(c, u0) + d(u0, u1) + d(u1, dest) ? 
        # Actually, destination is just a hint. 
        # Let's stick to the geometric function for consistency, it handles N=2 well.
        pass # Handled by general case below

    instance_id = _get_instance_id(distance_matrix)
    coords = _reconstruct_coords(distance_matrix, instance_id)
    
    return _select_next_node_geometric(current_node, destination_node, unvisited_nodes, distance_matrix, coords)

# EVOLVE-BLOCK-END
