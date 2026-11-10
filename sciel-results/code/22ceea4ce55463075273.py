"""Rewritten program: sphere covering with 50 points using convex-hull facet optimization."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 50

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
_DEADLINE = 140.0  # seconds
_start_time = None


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def _time_left():
    return _DEADLINE - _elapsed()


def _covering_radius(points):
    """
    Compute the covering radius in degrees for a set of points on the unit sphere.
    Uses the convex hull: the covering radius is the maximum circumradius of the
    spherical triangles (facets) of the convex hull.
    """
    points = np.asarray(points)
    n = len(points)
    if n < 4:
        return np.pi / 2  # fallback

    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]

    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2

    max_r = 0.0
    for simplex in hull.simplices:
        # simplex is a set of 3 indices for a tetrahedral facet
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        # Compute the circumradius of the spherical triangle
        # The circumcenter on the sphere is in the direction of a+b+c (if not degenerate)
        s = a + b + c
        norm_s = np.linalg.norm(s)
        if norm_s < 1e-10:
            # Degenerate: use the largest pairwise distance
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            r = max(d_ab, d_bc, d_ca) / 2.0
        else:
            s = s / norm_s
            # The circumradius is the angle from s to any vertex
            cos_r = np.clip(np.dot(s, a), -1, 1)
            r = np.arccos(cos_r)
        if r > max_r:
            max_r = r

    return max_r


def _covering_radius_degrees(points):
    return np.degrees(_covering_radius(points))


def _fibonacci_spiral(n=N):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_from_start(points0, time_budget=30.0):
    """
    Optimize a starting configuration using local optimization.
    We minimize the maximum facet circumradius.
    """
    points0 = np.array(points0, dtype=float)
    n = len(points0)

    # Flatten to 1D for optimizer
    x0 = points0.flatten()

    def objective(x):
        pts = x.reshape(n, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts = pts / norms
        return _covering_radius(pts)

    # Use Nelder-Mead with adaptive settings
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': 20000,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'adaptive': True,
        }
    )

    best_x = result.x
    best_val = result.fun

    # Normalize final result
    best_pts = best_x.reshape(n, 3)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_pts = best_pts / norms

    return best_pts, best_val


def construct():
    global _start_time
    _start_time = time.time()

    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_val = _covering_radius(points)

    # Try a few different initial configurations and optimize each
    initial_configs = [points]

    # Add a few perturbed versions
    rng = np.random.default_rng(42)
    for _ in range(3):
        perturbed = points + rng.normal(0, 0.1, size=(N, 3))
        # Normalize
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms
        initial_configs.append(perturbed)

    for i, config in enumerate(initial_configs):
        if _time_left() < 10:
            break

        # Allocate time per optimization
        time_budget = min(25.0, max(5.0, _time_left() - 5.0) / (len(initial_configs) - i))

        try:
            opt_points, opt_val = _optimize_from_start(config, time_budget=time_budget)
            if opt_val < best_val:
                best_val = opt_val
                best_points = opt_points
        except Exception:
            pass

    # Final polish with a more careful optimization if time allows
    if _time_left() > 15:
        try:
            final_points, final_val = _optimize_from_start(best_points, time_budget=_time_left() - 5)
            if final_val < best_val:
                best_val = final_val
                best_points = final_points
        except Exception:
            pass

    # Ensure output is valid: nonzero, unit-norm, finite
    best_points = np.array(best_points, dtype=float)
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_points = best_points / norms

    # Check for non-finite values
    if not np.all(np.isfinite(best_points)):
        # Fallback to Fibonacci spiral
        best_points = _fibonacci_spiral(N)

    return best_points
