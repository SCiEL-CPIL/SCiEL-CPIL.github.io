"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit
_CACHED = {}

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT

def _convex_hull_perimeter(points):
    """Compute perimeter of convex hull of points (array of shape (n, 2))."""
    if len(points) <= 2:
        if len(points) == 2:
            return 2.0 * np.linalg.norm(points[0] - points[1])
        return 0.0
    # Andrew's monotone chain
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        if len(pts) == 2:
            return 2.0 * np.linalg.norm(pts[0] - pts[1])
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.roll(hull, -1, axis=0) - hull, axis=1)))

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        n1, n2 = unvisited_nodes[0], unvisited_nodes[1]
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n1, destination_node]
        if cost1 <= cost2:
            return int(n1)
        else:
            return int(n2)

    # Time guard: fall back to nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Extract coordinates from distance matrix if available, else use index-based geometry
    # We need actual 2D coordinates for convex hull. Reconstruct from distance_matrix is hard.
    # Instead, use a geometric proxy: we'll use the distance matrix itself to build a
    # "geometric" score based on local density and hull-like expansion.
    #
    # Strategy: Score each candidate by:
    #   1. Distance from current (immediate cost)
    #   2. Distance to destination (future return cost)
    #   3. Local density penalty: how far the candidate is from its nearest unvisited neighbor
    #      (penalizes isolated points that would create long detours)
    #   4. Angle-based compactness: prefer nodes that continue the "direction" of travel
    #      to keep the tour compact (proxy for convex hull expansion)
    #
    # We use a weighted composite that balances these.

    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]

    # Local density: for each unvisited node, min distance to any other unvisited node
    if n_unvisited > 2:
        dist_subset = distance_matrix[unvisited_nodes][:, unvisited_nodes]
        np.fill_diagonal(dist_subset, np.inf)
        min_dist_to_other = np.min(dist_subset, axis=1)
    else:
        min_dist_to_other = np.ones(n_unvisited)

    # Normalize each component to [0, 1] range for fair weighting
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    norm_curr = normalize(dist_from_current)
    norm_dest = normalize(dist_to_dest)
    norm_density = normalize(min_dist_to_other)

    # Weighted composite score
    # Emphasize immediate distance and destination proximity, with density as a tiebreaker
    weights = np.array([1.0, 0.3, 0.4])
    scores = weights[0] * norm_curr + weights[1] * norm_dest + weights[2] * norm_density

    # Additional geometric compactness: penalize nodes that are far from the "center of mass"
    # of unvisited nodes (proxy for convex hull expansion)
    # Use distance matrix to estimate: nodes with low average distance to all unvisited are "central"
    avg_dist_to_unvisited = np.mean(dist_subset, axis=1) if n_unvisited > 2 else np.ones(n_unvisited)
    norm_center = normalize(avg_dist_to_unvisited)
    scores += 0.2 * norm_center

    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
