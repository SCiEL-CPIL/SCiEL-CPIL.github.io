# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
import itertools


def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return np.min(areas)


def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: A_min / A_hull."""
    a_hull = _hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_hull):
        return 0.0
    a_min = _min_triangle_area(points)
    if not np.isfinite(a_min) or a_min <= 0:
        return 0.0
    return a_min / a_hull


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 600.0  # safety margin before 630s kill

    # Start with a reasonable initial configuration
    # Use a grid-like arrangement as initial guess
    best_points = None
    best_score = 0.0

    # Try multiple random seeds and local improvements
    rng = np.random.default_rng(seed=42)

    for seed in range(50):
        if time.time() > deadline - 10:
            break

        if seed == 0:
            # Start with a structured arrangement: 13 points in a 4x4 grid minus corners
            # or a hexagonal packing
            points = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [0.5, 0.866],
                [1.5, 0.866],
                [2.5, 0.866],
                [0.0, 1.732],
                [1.0, 1.732],
                [2.0, 1.732],
                [3.0, 1.732],
                [1.5, 2.598],
                [2.5, 2.598]
            ])
        else:
            rng_local = np.random.default_rng(seed=seed)
            points = rng_local.random((n, 2))

        # Local optimization: simple hill climbing
        current_score = _score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()

        # Try simple local moves
        step_size = 0.1
        for _ in range(1000):
            if time.time() > deadline - 10:
                break

            # Pick a random point and move it
            i = np.random.randint(n)
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            new_points = points.copy()
            new_points[i] += [dx, dy]

            new_score = _score(new_points)
            if new_score > current_score:
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                # Decrease step size slowly
                step_size = max(0.01, step_size * 0.99)
            else:
                # Reset step size occasionally
                if np.random.random() < 0.05:
                    step_size = 0.1

    if best_points is not None:
        return best_points

    # Fallback: random points
    rng = np.random.default_rng(seed=42)
    return rng.random((n, 2))


# EVOLVE-BLOCK-END
