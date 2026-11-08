"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _normalize(points):
    """Normalize points to unit area convex hull."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return points
    scale = 1.0 / np.sqrt(hull_area)
    return points * scale


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin

    # Initial good configuration: 13 points in a roughly circular/regular arrangement
    # Start with a regular 13-gon, then perturb and optimize
    best_points = None
    best_score = -1.0

    # Generate initial points: regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_points = init_points.copy()
    best_score = _score(best_points)

    # Local optimization using gradient-free method (Nelder-Mead style or coordinate descent)
    # Use scipy.optimize.minimize with Nelder-Mead for speed
    from scipy.optimize import minimize

    def negative_score(params):
        points = params.reshape(N, 2)
        return -_score(points)

    # Multiple restarts with different perturbations
    rng = np.random.RandomState(42)

    for restart in range(50):
        if time.time() > deadline - 10.0:
            break

        # Create perturbed initial points
        if restart == 0:
            points = best_points.copy()
        else:
            # Mix of regular polygon and random perturbations
            base_angle = rng.uniform(0, 2 * np.pi)
            radii = 1.0 + rng.normal(0, 0.1, N)
            angles = base_angle + np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, 0.05, N)
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        # Run Nelder-Mead optimization
        try:
            result = minimize(
                negative_score,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.success or result.fun < -best_score:
                opt_points = result.x.reshape(N, 2)
                opt_score = -result.fun
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
