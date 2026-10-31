"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache to avoid recomputing coordinates from distance matrix
_coords_cache = {}
_dm_shape_cache = {}

def _get_coords(distance_matrix, n):
    """Reconstruct 2D coordinates from a Euclidean distance matrix using MDS-lite (SVD)."""
    key = id(distance_matrix)
    if key in _coords_cache:
        return _coords_cache[key]
    # Use classical MDS: double centering
    D = distance_matrix
    # Embed in 2D via SVD of the centered Gram matrix
    # B = -0.5 * J D^2 J
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * J @ D2 @ J
    # Eigen-decomposition (take top 2)
    evals, evecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    # Take top 2 positive eigenvalues
    k = 2
    positive = np.maximum(evals[:k], 0)
    X = evecs[:, :k] * np.sqrt(positive)
    _coords_cache[key] = X
    _dm_shape_cache[key] = n
    return X

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Spatial-sweep heuristic: pick the unvisited node whose angle (relative to
    current_node) is closest to the angle toward destination_node, with a
    small angular window. Ties broken by nearest distance.
    This encourages a consistent angular sweep, reducing backtracking.
    """
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])

    # Get coordinates (cached)
    total_n = distance_matrix.shape[0]
    coords = _get_coords(distance_matrix, total_n)

    cur = coords[current_node]
    dest = coords[destination_node]

    # Angle from current to destination
    dx_dest = dest[0] - cur[0]
    dy_dest = dest[1] - cur[1]
    target_angle = np.arctan2(dy_dest, dx_dest)

    # Angles from current to each unvisited
    unvisited_coords = coords[unvisited_nodes]
    dx = unvisited_coords[:, 0] - cur[0]
    dy = unvisited_coords[:, 1] - cur[1]
    angles = np.arctan2(dy, dx)

    # Angular difference (wrapped to [-pi, pi])
    angle_diff = np.abs(np.arctan2(np.sin(angles - target_angle), np.cos(angles - target_angle)))

    # Distance from current to each unvisited
    dists = np.sqrt(dx ** 2 + dy ** 2)

    # Primary sort: angular difference (smaller = more aligned with destination direction)
    # Secondary sort: distance (closer preferred among aligned candidates)
    # Use a combined score: angle_diff dominates, distance is a tiebreaker
    # Scale: angles in [0, pi], distances vary. Use a large multiplier for angle.
    # Actually, let's use a two-tier approach:
    # Tier 1: candidates within a small angular window (e.g., 60 degrees = pi/3)
    # Tier 2: among those, pick nearest. If none in window, pick minimum angle_diff.
    angle_window = np.pi / 3  # 60 degrees

    in_window = angle_diff <= angle_window
    if np.any(in_window):
        # Among in-window candidates, pick nearest
        windowed_dists = np.where(in_window, dists, np.inf)
        best_idx = np.argmin(windowed_dists)
    else:
        # No candidate in window; pick the one with smallest angle_diff, break by distance
        scores = angle_diff + 0.01 * dists
        best_idx = np.argmin(scores)

    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
