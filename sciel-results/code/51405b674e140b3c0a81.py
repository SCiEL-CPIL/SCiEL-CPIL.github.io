"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

# EVOLVE-BLOCK-START
def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(pts)
    min_area = np.inf
    # Vectorized triangle area computation
    # For each pair (i, j), compute cross product with all other points k
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    for i in range(n):
        for j in range(i + 1, n):
            # Vector from i to j
            v = pts[j] - pts[i]
            # Vectors from i to all other points
            w = pts - pts[i]
            # Cross product (2D): v_x * w_y - v_y * w_x
            cross = v[0] * w[:, 1] - v[1] * w[:, 0]
            areas = 0.5 * np.abs(cross)
            # Exclude i and j themselves (area 0)
            areas[i] = np.inf
            areas[j] = np.inf
            min_area = min(min_area, np.min(areas))
    return min_area


def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on sorted points
        # This is a rough fallback
        return 1.0


def _score(pts):
    """Score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp."""
    n = len(pts)
    log_areas = np.empty((n, n))
    log_areas.fill(-np.inf)
    
    for i in range(n):
        for j in range(i + 1, n):
            v = pts[j] - pts[i]
            w = pts - pts[i]
            cross = v[0] * w[:, 1] - v[1] * w[:, 0]
            areas = 0.5 * np.abs(cross)
            areas[i] = 0.0
            areas[j] = 0.0
            # Only consider positive areas
            valid = areas > 1e-15
            if np.any(valid):
                log_areas[i, j] = np.log(np.sum(np.exp(beta * areas[valid]))) / beta
    
    # The soft min over all pairs
    max_log = np.max(log_areas)
    if max_log == -np.inf:
        return 0.0
    soft_min = max_log - np.log(np.sum(np.exp(log_areas - max_log))) / beta
    return soft_min


def _soft_score(pts, beta):
    """Soft score: soft min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    soft_min = _soft_min_area(pts, beta)
    return soft_min / hull_area


def _regular_polygon(n, radius, phase):
    """Generate n points on a regular polygon with given radius and phase."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def _construct_multi_ring(params):
    """
    Construct 14 points as union of concentric regular polygons.
    params: [r1, r2, r3, phase1, phase2, phase3]
    Ring 1: 7 points, Ring 2: 5 points, Ring 3: 2 points
    """
    r1, r2, r3, phase1, phase2, phase3 = params
    pts1 = _regular_polygon(7, r1, phase1)
    pts2 = _regular_polygon(5, r2, phase2)
    pts3 = _regular_polygon(2, r3, phase3)
    return np.vstack([pts1, pts2, pts3])


def _score_multi_ring(params):
    """Negative score for optimization (we maximize score, so minimize negative)."""
    pts = _construct_multi_ring(params)
    return -_score(pts)


def _soft_score_multi_ring(params, beta):
    """Soft negative score for optimization."""
    pts = _construct_multi_ring(params)
    return -_soft_score(pts, beta)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_pts = None
    
    # Initial guess: regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_init = np.column_stack((np.cos(angles), np.sin(angles)))
    best_pts = pts_init.copy()
    best_score = _score(pts_init)
    
    # Differential evolution over multi-ring parameters
    # params: [r1, r2, r3, phase1, phase2, phase3]
    # r1 in [0.5, 1.5], r2 in [0.2, 1.0], r3 in [0.0, 0.5]
    # phases in [0, 2*pi]
    bounds = [
        (0.3, 1.5),  # r1
        (0.1, 1.0),  # r2
        (0.0, 0.5),  # r3
        (0, 2 * np.pi),  # phase1
        (0, 2 * np.pi),  # phase2
        (0, 2 * np.pi),  # phase3
    ]
    
    # Try differential evolution
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            result = differential_evolution(
                _score_multi_ring,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=True,
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = _construct_multi_ring(result.x)
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Local refinement with L-BFGS-B using soft min
        for beta in [50, 200, 1000, 5000]:
            if time.time() - start_time > deadline - 15:
                break
            try:
                # Start from current best
                x0 = np.array([1.0, 0.6, 0.2, 0.0, 0.0, 0.0])
                res = minimize(
                    lambda x: _soft_score_multi_ring(x, beta),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
                )
                if res.fun < -best_score:
                    best_score = -res.fun
                    best_pts = _construct_multi_ring(res.x)
            except Exception:
                pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try a few random perturbations of the best point and polish
        for trial in range(5):
            if time.time() - start_time > deadline - 5:
                break
            # Perturb
            perturbation = np.random.randn(6) * 0.1
            x_pert = np.array([1.0, 0.6, 0.2, 0.0, 0.0, 0.0]) + perturbation
            x_pert = np.clip(x_pert, [b[0] for b in bounds], [b[1] for b in bounds])
            try:
                res = minimize(
                    lambda x: _soft_score_multi_ring(x, 5000),
                    x_pert,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                if res.fun < -best_score:
                    best_score = -res.fun
                    best_pts = _construct_multi_ring(res.x)
            except Exception:
                pass
    
    # Final: also try full 28D local optimization from best point
    elapsed = time.time() - start_time
    if elapsed < deadline - 5 and best_pts is not None:
        try:
            # Fix p0 at origin, p1 at (1,0) to remove translation/rotation
            # But we need to be careful: our best_pts may not have this property
            # Just do a local polish in the full space
            def full_score(x):
                pts = x.reshape(N, 2)
                return -_score(pts)
            
            x0 = best_pts.flatten()
            res = minimize(
                full_score,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10},
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
