"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """Calculate minimum triangle area using vectorized cross products."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            if j + 1 < n:
                rest = points[j+1:]
                v2 = rest - pi
                cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
                areas = 0.5 * np.abs(cross)
                if areas.size > 0:
                    local_min = np.min(areas)
                    if local_min < min_area:
                        min_area = local_min
    return min_area

def _convex_hull_area_fast(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the normalized minimum triangle area."""
    hull_area = _convex_hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri_area = _min_triangle_area_fast(points)
    return min_tri_area / hull_area

def _neg_score_coords(points_flat):
    """Objective for direct coordinate optimization."""
    points = points_flat.reshape(N, 2)
    return -_score(points)

def _regular_polygon(n):
    """Return vertices of regular n-gon centered at origin with circumradius 1."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angles), np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Initialize with regular 14-gon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Phase 1: Quick differential evolution on 28D coordinates
    # Use a bounded box around the unit circle
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    def _time_check():
        return time.time() > deadline - 5
    
    try:
        de_result = differential_evolution(
            _neg_score_coords,
            bounds,
            maxiter=80,
            popsize=12,
            tol=1e-5,
            mutation=(0.4, 0.8),
            recombination=0.7,
            seed=42,
            polish=False,
            init='sobol',
            callback=lambda xk, convergence: None,
            updating='immediate'
        )
        if de_result.fun < -best_score:
            candidate_points = de_result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points.copy()
                best_score = candidate_score
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead polish on coordinates from best DE result
    if time.time() < deadline - 10:
        try:
            nm_result = minimize(
                _neg_score_coords,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            candidate_points = nm_result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points.copy()
                best_score = candidate_score
        except Exception:
            pass
    
    # Phase 3: Multiple Nelder-Mead restarts from perturbed regular polygons
    if time.time() < deadline - 10:
        np.random.seed(123)
        for restart in range(6):
            if time.time() > deadline - 10:
                break
            
            # Create perturbed starting point
            base = _regular_polygon(N)
            noise = np.random.normal(0, 0.15, (N, 2))
            init_points = base + noise
            init_flat = init_points.flatten()
            
            try:
                nm_result = minimize(
                    _neg_score_coords,
                    init_flat,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                candidate_points = nm_result.x.reshape(N, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points.copy()
                    best_score = candidate_score
            except Exception:
                pass
    
    # Phase 4: Final polish if time allows
    if time.time() < deadline - 5:
        try:
            nm_result = minimize(
                _neg_score_coords,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            candidate_points = nm_result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points.copy()
                best_score = candidate_score
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
