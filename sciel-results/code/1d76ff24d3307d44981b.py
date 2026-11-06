"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Hard deadline: 140 seconds (10s safety margin for 150s kill)
DEADLINE = 140.0


def _covering_radius_rad(points):
    """
    Compute the covering radius (in radians) of points on the unit sphere.
    The covering radius is the maximum over all y on the sphere of min_i ang_dist(y, p_i).
    This is equivalent to the maximum circumradius of the spherical caps at the facets
    of the convex hull of the points.
    """
    # Normalize points
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms

    # Compute convex hull to find facets
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use all pairwise distances
        dots = np.clip(points @ points.T, -1.0, 1.0)
        dists = np.arccos(dots)
        return np.max(dists)

    # For each facet (triangle), compute the circumradius
    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]

        # Compute side lengths (angular distances)
        ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
        ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))

        # Compute normal to the plane
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            # Degenerate triangle
            r = max(ab, bc, ca) / 2
            max_r = max(max_r, r)
            continue
        n = n / n_norm

        # Use the formula for circumradius of a spherical triangle
        cos_a, cos_b, cos_c = np.cos(ab), np.cos(bc), np.cos(ca)

        numerator = cos_a * cos_b * cos_c - cos_a - cos_b - cos_c + 1
        denominator = 2 * (1 - cos_a - cos_b - cos_c + cos_a * cos_b + cos_b * cos_c + cos_c * cos_a)

        if abs(denominator) < 1e-10:
            # Degenerate case
            r = max(ab, bc, ca) / 2
            max_r = max(max_r, r)
            continue

        cos_2r = numerator / denominator
        cos_2r = np.clip(cos_2r, -1.0, 1.0)
        r = 0.5 * np.arccos(cos_2r)

        max_r = max(max_r, r)

    return max_r


def _smooth_objective(x_flat, query_points, beta):
    """
    Smooth approximation of the covering radius using log-sum-exp over query points.
    """
    x = x_flat.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    x = x / norms

    # Compute squared chordal distances between query points and config points
    cos_dists = query_points @ x.T  # Shape (M, N)
    cos_dists = np.clip(cos_dists, -1.0, 1.0)
    chordal_dists_sq = 2.0 * (1.0 - cos_dists)

    # Find min chordal dist sq for each query point
    min_dists_sq = np.min(chordal_dists_sq, axis=1)

    # Log-sum-exp of beta * min_dists_sq
    scaled = beta * min_dists_sq
    max_scaled = np.max(scaled)
    lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))

    return lse / beta


def _raw_objective(x_flat):
    """
    Raw covering radius objective.
    """
    x = x_flat.reshape(N, 3)
    return _covering_radius_rad(x)


def construct():
    start_time = time.time()

    # Start from Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_points = points.copy()
    best_radius = _covering_radius_rad(best_points)

    # Generate random query points for smooth objective
    np.random.seed(42)
    M = 2000
    u = np.random.uniform(-1, 1, (M, 3))
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u = u / norms

    # Phase 1: Coarse optimization with smooth objective
    try:
        result = minimize(
            lambda x: _smooth_objective(x, u, beta=10.0),
            points.flatten(),
            method='Powell',
            options={'maxiter': 3000, 'maxfev': 3000, 'disp': False}
        )

        if result.x is not None:
            opt_points = result.x.reshape(N, 3)
            opt_radius = _covering_radius_rad(opt_points)
            if opt_radius < best_radius:
                best_points = opt_points
                best_radius = opt_radius
    except Exception:
        pass

    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 30:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return best_points / norms

    # Phase 2: Fine optimization with raw objective
    try:
        result = minimize(
            _raw_objective,
            best_points.flatten(),
            method='Powell',
            options={'maxiter': 2000, 'maxfev': 2000, 'disp': False}
        )

        if result.x is not None:
            opt_points = result.x.reshape(N, 3)
            opt_radius = _covering_radius_rad(opt_points)
            if opt_radius < best_radius:
                best_points = opt_points
                best_radius = opt_radius
    except Exception:
        pass

    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return best_points / norms

    # Phase 3: Local search with perturbations
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break

        # Perturb the best solution
        noise = np.random.randn(N, 3) * 0.05
        perturbed = best_points + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms

        try:
            result = minimize(
                _raw_objective,
                perturbed.flatten(),
                method='Powell',
                options={'maxiter': 500, 'maxfev': 500, 'disp': False}
            )

            if result.x is not None:
                opt_points = result.x.reshape(N, 3)
                opt_radius = _covering_radius_rad(opt_points)
                if opt_radius < best_radius:
                    best_points = opt_points
                    best_radius = opt_radius
        except Exception:
            pass

    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms

    return best_points
# EVOLVE-BLOCK-END
