# EVOLVE-BLOCK-START
import numpy as np
import time

# Wall-clock deadline guard with safety margin (evaluator kills at 150s; we target 120s).
_DEADLINE = 115.0  # seconds from process start; safety margin before hard limit
_START = time.time()


def _circumradius_score(current_node, destination_node, unvisited_nodes, distance_matrix,
                        coords_cache):
    """
    Return an array of geometric penalties (circumradius) for each candidate in
    unvisited_nodes, given the triangle (current_node, destination_node, candidate).

    Uses cached 2D/3D coordinates extracted lazily from the distance matrix via
    a lightweight embedding (we assume Euclidean; recover coordinates by
    multi-dimensional scaling is too slow, so we instead compute circumradius
    directly from pairwise distances using the formula:
        R = (a * b * c) / (4 * Area)
    where a, b, c are side lengths and Area via Heron's formula.
    """
    n = len(unvisited_nodes)
    if n == 0:
        return np.zeros(0)

    # Side lengths of triangle:
    #   a = dist(current, destination)  (fixed)
    #   b = dist(current, candidate)    (varies)
    #   c = dist(destination, candidate) (varies)
    a = distance_matrix[current_node, destination_node]
    b = distance_matrix[current_node, unvisited_nodes]  # (n,)
    c = distance_matrix[destination_node, unvisited_nodes]  # (n,)

    # Degenerate: if a == 0 (current == destination), circumradius undefined -> 0.
    if a < 1e-12:
        return np.zeros(n)

    # Heron's formula for area:
    s = 0.5 * (a + b + c)
    # Guard against tiny/negative due to floating point or degenerate triangles.
    area_sq = s * (s - a) * (s - b) * (s - c)
    area_sq = np.maximum(area_sq, 0.0)
    area = np.sqrt(area_sq)

    # Circumradius R = (a * b * c) / (4 * area)
    # Where area ~ 0 (collinear), R -> inf; we cap it to avoid blow-up.
    denom = 4.0 * area
    with np.errstate(divide='ignore', invalid='ignore'):
        R = (a * b * c) / denom
    R = np.where(area < 1e-12, 0.0, R)  # collinear -> no penalty
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return R


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard: if we're close to the limit, use the cheapest safe choice.
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        # Pure nearest neighbor from current_node as a safe fallback.
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n = len(unvisited_nodes)
    if n == 0:
        # No unvisited nodes; return current as a safe default (shouldn't happen in valid tours).
        return int(current_node)
    if n == 1:
        return int(unvisited_nodes[0])

    # Edge distance from current to each candidate.
    edge = distance_matrix[current_node, unvisited_nodes]  # (n,)

    # Geometric circumradius penalty.
    R = _circumradius_score(current_node, destination_node, unvisited_nodes, distance_matrix, None)

    # Blend: primary is edge distance; secondary is circumradius (tight turns).
    # Normalize R to the scale of edge distances so alpha is interpretable.
    R_max = np.max(R) if n > 0 else 1.0
    edge_max = np.max(edge) if n > 0 else 1.0
    if R_max > 1e-12 and edge_max > 1e-12:
        R_norm = R * (edge_max / R_max)
    else:
        R_norm = R

    alpha = 0.5  # weight of geometric term
    score = edge + alpha * R_norm

    # Tie-breaker: prefer candidates closer to destination (helps converge).
    dest_edge = distance_matrix[destination_node, unvisited_nodes]  # (n,)
    dest_max = np.max(dest_edge) if n > 0 else 1.0
    if dest_max > 1e-12:
        score += 0.05 * dest_edge / dest_max

    idx = int(np.argmin(score))
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
