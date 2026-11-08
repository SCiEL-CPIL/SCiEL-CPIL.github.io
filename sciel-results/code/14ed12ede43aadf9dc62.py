"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        if len(points) < 3:
            return 0.0
        # Fallback: compute area using shoelace formula on angle-sorted points
        c = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
        order = np.argsort(angles)
        pts = points[order]
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) * 0.5


def _score(points):
    """Score = minimum triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _regular_polygon(n):
    """Generate vertices of a regular n-gon on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))


def construct():
    start_time = time.time()
    # Use a safety margin. The limit is 150s. We stop at 145s.
    deadline = start_time + 145.0

    # Initial guess: Regular polygon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)

    rng = np.random.default_rng(42)

    # Objective function for optimization
    # We want to maximize min_triangle_area / hull_area
    # So we minimize -min_triangle_area / hull_area
    def objective(x):
        pts = x.reshape(-1, 2)
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-10:
            return 1e10
        min_area = _min_triangle_area(pts)
        return -min_area / hull_area

    # Phase 1: Nelder-Mead optimization from multiple starts
    # We try several random perturbations of the regular polygon
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break

        if restart == 0:
            points0 = best_points.copy()
        else:
            # Generate a random perturbation
            # Small perturbations to stay in the same "basin" initially
            noise_scale = 0.05 + 0.05 * rng.random()
            points0 = _regular_polygon(N) + rng.normal(0, noise_scale, size=(N, 2))

        # Run Nelder-Mead
        # maxiter is set conservatively to ensure we don't hang
        # Nelder-Mead is derivative-free and robust
        res = minimize(objective, points0.flatten(), method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})

        if time.time() > deadline - 5.0:
            break

        candidate = res.x.reshape(-1, 2)
        cand_score = _score(candidate)

        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate

    # Phase 2: High-precision polish
    # Take the best point found and run a tighter optimization
    if time.time() < deadline - 10.0:
        def objective_tight(x):
            pts = x.reshape(-1, 2)
            hull_area = _convex_hull_area(pts)
            if hull_area < 1e-10:
                return 1e10
            min_area = _min_triangle_area(pts)
            return -min_area / hull_area

        # Try BFGS or L-BFGS-B if we have gradients? No, we don't have analytic gradients easily.
        # Stick with Nelder-Mead but with tighter tolerances and more iterations on the best point.
        # Also try a few more random restarts with very small noise around the best point.
        for polish in range(10):
            if time.time() > deadline - 2.0:
                break

            if polish == 0:
                points0 = best_points.copy()
            else:
                points0 = best_points + rng.normal(0, 0.01, size=(N, 2))

            res = minimize(objective_tight, points0.flatten(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})

            if time.time() > deadline - 2.0:
                break

            candidate = res.x.reshape(-1, 2)
            cand_score = _score(candidate)

            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate

    return best_points
# EVOLVE-BLOCK-END
