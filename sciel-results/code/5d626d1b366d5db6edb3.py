# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _vertex_score(distances: np.ndarray) -> float:
    """Compute the score for a single vertex given sorted distances to other 19 vertices."""
    d = np.sort(distances)
    if d[0] < 0.05:
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)


def _min_vertex_score(points: np.ndarray) -> float:
    """Compute the minimum vertex score over all vertices."""
    n = len(points)
    scores = []
    for i in range(n):
        diffs = points - points[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists = np.delete(dists, i)  # exclude self
        scores.append(_vertex_score(dists))
    return min(scores)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon optimized to maximize the minimum vertex equidistance score.
    Uses a parameterization by angular offsets from the regular polygon and optimizes
    via scipy to find a shape where every vertex has a well-clustered distance spectrum
    at the critical boundary.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    def objective(params: np.ndarray) -> float:
        # params: 20 angular perturbations
        angles = base_angles + params
        # Ensure angles are in [0, 2π) and sorted for convexity
        angles = np.sort(angles) % (2.0 * np.pi)
        # If angles wrapped, re-sort
        angles = np.sort(angles)
        # Check for very close angles (non-convex or degenerate)
        gaps = np.diff(np.concatenate([angles, [angles[0] + 2.0 * np.pi]]))
        if np.min(gaps) < 0.01:
            return -1.0  # penalty for near-degenerate
        x = np.cos(angles)
        y = np.sin(angles)
        points = np.stack([x, y], axis=1)
        # Normalize: subtract centroid, divide by mean distance to centroid
        centroid = np.mean(points, axis=0)
        points = points - centroid
        dists_to_centroid = np.linalg.norm(points, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return -1.0
        points = points / mean_dist
        # Check convexity: all cross products should have the same sign
        cross = []
        for i in range(n):
            p0 = points[i]
            p1 = points[(i + 1) % n]
            p2 = points[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross.append(v1[0] * v2[1] - v1[1] * v2[0])
        if len(set(np.sign(cross))) > 1:
            return -1.0  # not strictly convex
        score = _min_vertex_score(points)
        return -score  # minimize negative score

    # Start from zero perturbation (regular polygon)
    x0 = np.zeros(n)
    # Use a small initial perturbation to help optimizer
    np.random.seed(42)
    x0 = np.random.normal(0, 0.01, n)

    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4,
                               'adaptive': True})

    best_params = result.x
    best_angles = np.sort(base_angles + best_params) % (2.0 * np.pi)
    best_angles = np.sort(best_angles)

    x = np.cos(best_angles)
    y = np.sin(best_angles)
    points = np.stack([x, y], axis=1)

    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist > 0.01:
        points = points / mean_dist

    return points


# EVOLVE-BLOCK-END
