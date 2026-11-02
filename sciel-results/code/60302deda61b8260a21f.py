"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull


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
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective(x):
    """Objective for minimization: negative of the score."""
    points = x.reshape(-1, 2)
    s = _score(points)
    return -s


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Initial good starting point: regular 14-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    # Normalize to unit area
    init_hull_area = _convex_hull_area(init_points)
    if init_hull_area > 1e-12:
        init_points = init_points / np.sqrt(init_hull_area)

    best_x = init_points.ravel()
    best_score = _score(init_points)

    # Differential evolution over the 28 coordinates
    # Bounds: points within [-1, 1] x [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)

    try:
        result = differential_evolution(
            _objective,
            bounds,
            x0=best_x,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            init='latinhypercube',
            callback=lambda xk, convergence: (
                None  # no early stop callback needed; rely on maxiter and time
            )
        )
        if result.success and result.fun < -best_score:
            best_x = result.x
            best_score = -result.fun
    except Exception:
        pass

    # Check time; if we have time left, try a few more random restarts with local polish
    if time.time() < deadline - 5.0:
        rng = np.random.RandomState(42)
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            # Random perturbation of the current best
            perturbed = best_x + rng.randn(2 * N) * 0.1
            # Clamp to bounds
            perturbed = np.clip(perturbed, -1.0, 1.0)
            # Quick local search via a few Nelder-Mead steps
            from scipy.optimize import minimize
            try:
                res = minimize(
                    _objective,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < -best_score:
                    best_x = res.x
                    best_score = -res.fun
            except Exception:
                pass

    best_points = best_x.reshape(-1, 2)
    return best_points
# EVOLVE-BLOCK-END
