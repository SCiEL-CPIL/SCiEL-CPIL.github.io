"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute minimum triangle area over all C(n,3) triples."""
    n = len(points)
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute area of convex hull of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, .volume gives area
    except Exception:
        return 0.0


def _score(points):
    """Score = min_triangle_area / convex_hull_area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective(x, n_points=N):
    """Negative score for minimization."""
    points = x.reshape(n_points, 2)
    return -_score(points)


def _generate_grid_config():
    """Generate a grid-like configuration for 13 points."""
    # 4x4 grid minus one corner, slightly perturbed
    pts = []
    for i in range(4):
        for j in range(4):
            if len(pts) < N:
                pts.append([i / 3.0, j / 3.0])
    return np.array(pts)


def _generate_quasirandom_config():
    """Generate points using a low-discrepancy sequence (Halton-like)."""
    def halton(index, base):
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    pts = []
    for idx in range(1, N + 1):
        x = halton(idx, 2)
        y = halton(idx, 3)
        pts.append([x, y])
    return np.array(pts)


def _generate_circular_config():
    """Generate points on a circle with some interior points."""
    pts = []
    # 7 points on outer circle
    for i in range(7):
        angle = 2 * np.pi * i / 7
        pts.append([0.5 + 0.45 * np.cos(angle), 0.5 + 0.45 * np.sin(angle)])
    # 6 points in inner region
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        r = 0.25
        pts.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    return np.array(pts)


def _generate_random_config(seed):
    """Generate random points in unit square."""
    rng = np.random.RandomState(seed)
    return rng.random((N, 2))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_points = None
    best_score = -np.inf

    # Generate multiple initial configurations
    initial_configs = [
        _generate_grid_config(),
        _generate_quasirandom_config(),
        _generate_circular_config(),
        _generate_random_config(42),
        _generate_random_config(123),
    ]

    # Evaluate initial configurations
    for config in initial_configs:
        if time.time() > deadline:
            break
        s = _score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()

    # Local optimization from best initial config and a few others
    configs_to_optimize = [initial_configs[0], initial_configs[1], initial_configs[2], best_points]
    # Add some random perturbations of best
    rng = np.random.RandomState(999)
    for _ in range(3):
        perturbed = best_points + 0.05 * rng.random((N, 2))
        # Keep in [0,1]
        perturbed = np.clip(perturbed, 0.0, 1.0)
        configs_to_optimize.append(perturbed)

    for config in configs_to_optimize:
        if time.time() > deadline - 5.0:
            break

        x0 = config.flatten()
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            if time.time() > deadline:
                break

            opt_points = result.x.reshape(N, 2)
            s = _score(opt_points)
            if s > best_score:
                best_score = s
                best_points = opt_points.copy()
        except Exception:
            continue

    # If we still have time, try a few more random restarts with local opt
    if time.time() < deadline - 10.0 and best_points is not None:
        rng2 = np.random.RandomState(777)
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            x0 = rng2.random(N * 2)
            try:
                result = minimize(
                    _objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
                if time.time() > deadline:
                    break
                opt_points = result.x.reshape(N, 2)
                s = _score(opt_points)
                if s > best_score:
                    best_score = s
                    best_points = opt_points.copy()
            except Exception:
                continue

    if best_points is None:
        # Fallback: return the best initial config or a default
        if initial_configs:
            return initial_configs[0]
        return np.random.random((N, 2))

    return best_points
# EVOLVE-BLOCK-END
