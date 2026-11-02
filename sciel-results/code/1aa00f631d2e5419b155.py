"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective(params, N):
    """
    Objective function for optimization.
    params: array of 2*N coordinates (flattened)
    Returns negative score (we minimize negative score to maximize score).
    """
    points = params.reshape(N, 2)
    # Penalize if points are not in convex position or degenerate
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            return 1.0  # Bad: not all points on convex hull
    except Exception:
        return 1.0
    return -_score(points)


def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_points = None
    best_score = -1.0

    # Start with regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_points = init_points.copy()
    best_score = _score(best_points)

    # Try multiple random perturbations and local optimizations
    rng = np.random.RandomState(42)

    for restart in range(30):
        if time.time() > deadline - 5:
            break

        # Generate initial guess: regular polygon with random perturbation
        if restart == 0:
            x0 = init_points.flatten()
        else:
            angles = 2 * np.pi * np.arange(N) / N + rng.uniform(-0.1, 0.1, N)
            radii = 1.0 + rng.uniform(-0.1, 0.1, N)
            x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()

        # Use L-BFGS-B for local optimization
        try:
            result = minimize(
                _objective,
                x0,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if result.success:
                points = result.x.reshape(N, 2)
                score = _score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        except Exception:
            pass

    # Final refinement with more iterations on the best point
    if best_points is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if result.success:
                points = result.x.reshape(N, 2)
                score = _score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
