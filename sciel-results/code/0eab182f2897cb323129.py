"""Rewritten program for Heilbronn 13-point problem: maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _score(pts):
    """Compute min triangle area divided by convex hull area. Returns 0 if invalid."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    if n < 3:
        return 0.0
    # Convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # in 2D, volume is area
    except Exception:
        return 0.0
    if hull_area <= 1e-12:
        return 0.0
    # Compute all triangle areas vectorized
    # For each pair (i,j), compute cross product with all k
    min_area = float('inf')
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            diff1 = pj - pi
            # Cross product of diff1 with (pk - pi) for all k > j
            for k in range(j + 1, n):
                pk = pts[k]
                diff2 = pk - pi
                cross = abs(diff1[0] * diff2[1] - diff1[1] * diff2[0])
                tri_area = 0.5 * cross
                if tri_area < min_area:
                    min_area = tri_area
    return min_area / hull_area


def _neg_score(pts_flat):
    """Negative score for minimization. Reshapes flat array to (N,2)."""
    pts = pts_flat.reshape(N, 2)
    s = _score(pts)
    if s <= 0:
        return 1e6  # large penalty for invalid
    return -s


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _concentric_polygons(n_outer, n_inner, r_outer=1.0, r_inner=0.5, offset=0.0):
    """Generate concentric polygons."""
    outer = _regular_polygon(n_outer, r_outer)
    angles_inner = 2 * np.pi * np.arange(n_inner) / n_inner + offset
    inner = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    return np.vstack([outer, inner])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Generate multiple initial configurations
    initial_configs = []

    # 1. Regular 13-gon
    initial_configs.append(_regular_polygon(N, 1.0))

    # 2. Concentric: 7 outer + 6 inner
    initial_configs.append(_concentric_polygons(7, 6, 1.0, 0.5, np.pi/6))

    # 3. Concentric: 12 outer + 1 center
    outer12 = _regular_polygon(12, 1.0)
    center = np.array([[0.0, 0.0]])
    initial_configs.append(np.vstack([outer12, center]))

    # 4. Concentric: 9 outer + 4 inner
    initial_configs.append(_concentric_polygons(9, 4, 1.0, 0.4, np.pi/9))

    # 5. Concentric: 6 outer + 7 inner
    initial_configs.append(_concentric_polygons(6, 7, 1.0, 0.6, np.pi/7))

    # 6. Regular 13-gon with slight radial perturbation
    rng = np.random.default_rng(42)
    angles = 2 * np.pi * np.arange(N) / N
    radii = 1.0 + 0.15 * rng.standard_normal(N)
    initial_configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))

    # 7. Grid-like: 4x4 grid minus one corner, adjusted
    # Actually let's do a 3x5 grid with some adjustment
    # Let's try a hexagonal packing inspired config
    # For now, add a few more concentric variants with different radii
    initial_configs.append(_concentric_polygons(7, 6, 1.0, 0.6, np.pi/6))
    initial_configs.append(_concentric_polygons(7, 6, 1.0, 0.4, 0.0))
    initial_configs.append(_concentric_polygons(13, 0, 1.0, 0.0, 0.0))  # same as regular

    # Evaluate all initial configs
    best_pts = None
    best_score = 0.0
    for cfg in initial_configs:
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_pts = cfg.copy()

    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_pts

    # Now do Nelder-Mead optimization from multiple starting points
    # We'll try to optimize from the top few initial configs
    config_scores = [( _score(cfg), cfg) for cfg in initial_configs]
    config_scores.sort(key=lambda x: -x[0])

    # Try Nelder-Mead from top 5 configs (or fewer if time runs out)
    max_starts = 5
    for idx in range(min(max_starts, len(config_scores))):
        if time.time() > deadline - 5:
            break

        cfg_score, cfg = config_scores[idx]
        x0 = cfg.flatten()

        # Nelder-Mead options
        options = {
            'maxiter': 200,
            'xatol': 1e-6,
            'fatol': 1e-8,
            'adaptive': True,
        }

        try:
            result = minimize(
                _neg_score,
                x0,
                method='Nelder-Mead',
                options=options
            )
            if result.success or result.fun < -best_score:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass

        elapsed = time.time() - start_time
        if elapsed > deadline:
            break

    # If we still have time, do a second round with more iterations
    # from the best found so far
    if time.time() < deadline - 10 and best_pts is not None:
        x0 = best_pts.flatten()
        options2 = {
            'maxiter': 300,
            'xatol': 1e-7,
            'fatol': 1e-9,
            'adaptive': True,
        }
        try:
            result2 = minimize(
                _neg_score,
                x0,
                method='Nelder-Mead',
                options=options2
            )
            new_pts = result2.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
