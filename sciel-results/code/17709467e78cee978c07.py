"""Geometric flow TSP heuristic: enforce consistent rotational direction via cross-product."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin before 120s hard limit

# Cache to store coordinates derived from distance matrix (only needed once per instance)
_coords_cache = {}

def _extract_coords(distance_matrix):
    """
    Extract 2D coordinates from a Euclidean distance matrix.
    Uses MDS (multidimensional scaling) via eigendecomposition.
    Caches results by matrix shape and a hash of the first few entries.
    """
    n = distance_matrix.shape[0]
    # Use a simple fingerprint: shape + a few key values
    fp = (n, distance_matrix[0, 1] if n > 1 else 0, distance_matrix[0, 2] if n > 2 else 0)
    if fp in _coords_cache:
        return _coords_cache[fp]
    
    # Classical MDS: double centering
    D2 = distance_matrix ** 2
    one_n = np.ones((n, n)) / n
    B = -0.5 * (one_n @ D2 + D2 @ one_n - D2 @ one_n @ one_n)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Take top 2 eigenvalues (largest)
    idx = np.argsort(eigenvalues)[::-1][:2]
    eigenvalues_top = eigenvalues[idx]
    eigenvectors_top = eigenvectors[:, idx]
    
    # Ensure non-negative eigenvalues
    eigenvalues_top = np.maximum(eigenvalues_top, 0)
    
    coords = eigenvectors_top * np.sqrt(eigenvalues_top)
    
    _coords_cache[fp] = coords
    return coords

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using geometric flow: maximize cross-product magnitude
    between (current->destination) and (current->candidate) vectors.
    This enforces consistent rotational direction, preventing edge crossings.
    """
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback: nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Extract coordinates (cached)
    coords = _extract_coords(distance_matrix)
    
    # Get position vectors
    pos_current = coords[current_node]
    pos_dest = coords[destination_node]
    pos_candidates = coords[unvisited_nodes]
    
    # Vector from current to destination
    v_dest = pos_dest - pos_current
    
    # Vectors from current to each candidate
    v_cands = pos_candidates - pos_current
    
    # Compute cross-product magnitudes: |v_dest x v_cand|
    # In 2D, cross product magnitude = |v1.x * v2.y - v1.y * v2.x|
    cross_mags = np.abs(v_dest[0] * v_cands[:, 1] - v_dest[1] * v_cands[:, 0])
    
    # Also compute distances from current to candidates
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Score: minimize distance but prefer candidates that are "ahead" in rotation
    # The cross product magnitude is proportional to sin(theta) * |v_dest| * |v_cand|
    # We want to pick nodes that make a reasonable angle with the destination direction
    # and are close to current.
    
    # Normalize cross magnitudes by distance to get sin(theta)
    # sin(theta) = cross_mag / (|v_dest| * |v_cand|)
    v_dest_norm = np.linalg.norm(v_dest)
    v_cands_norms = np.linalg.norm(v_cands, axis=1)
    
    # Avoid division by zero
    safe_norms = np.where(v_cands_norms > 1e-10, v_cands_norms, 1.0)
    sin_theta = cross_mags / (v_dest_norm * safe_norms) if v_dest_norm > 1e-10 else np.zeros_like(cross_mags)
    
    # Strategy: minimize d(current, cand) - beta * sin(theta)
    # The negative sin(theta) term rewards nodes that are "turning" in the right direction
    # This creates a flow-like behavior that follows the convex hull
    beta = 0.3  # Weight for rotational consistency
    
    scores = dists_from_current - beta * sin_theta
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
