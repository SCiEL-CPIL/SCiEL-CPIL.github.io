"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

def _all_triangle_areas(pts):
    """Vectorized areas of all C(n,3) triangles."""
    n = len(pts)
    parts = []
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            v2 = pts[j + 1:] - pts[i]
            if len(v2) == 0:
                continue
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            parts.append(0.5 * cross)
    return np.concatenate(parts)

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _convex_hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_area = np.min(_all_triangle_areas(pts))
    return min_area / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _soft_min_triangle_area(pts, alpha=50.0):
    """
    Smooth approximation of the minimum triangle area.
    Uses -log(sum(exp(-alpha * areas))) / alpha
    This is differentiable and guides optimizers better.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    log_sum = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                min_a = np.min(areas)
                if min_a > 0:
                    shifted = -alpha * (areas - min_a)
                    log_sum_partial = np.log(np.sum(np.exp(shifted))) - alpha * min_a
                    if log_sum_partial > log_sum:
                        log_sum = log_sum_partial
    if log_sum == -np.inf:
        return 0.0
    return -log_sum / alpha

def _hull_area_safe(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 1.0

def _smooth_score(pts):
    """Smooth score surrogate."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area_safe(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _soft_min_triangle_area(pts, alpha=50.0) / hull_a

def _neg_smooth_score(params):
    pts = params.reshape(-1, 2)
    return -_smooth_score(pts)

def _make_symmetric_points(params, config):
    """
    Generate points from symmetric parameters.
    config: list of (n_points, radius, angle_offset) for each ring
    params: [r1, a1, r2, a2, ...] where ri is radius, ai is angular offset
    """
    pts = []
    idx = 0
    for n_pts, base_radius, base_offset in config:
        if n_pts == 0:
            continue
        r = params[idx]
        a = params[idx + 1]
        idx += 2
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + a
        ring = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts.append(ring)
    if not pts:
        return np.zeros((N, 2))
    return np.vstack(pts)

def _symmetric_configs():
    """Generate various symmetric configurations for 13 points."""
    configs = []
    
    # Single ring: 13 points
    configs.append([(13, 1.0, 0.0)])
    
    # Two rings: various splits
    for n1 in range(3, 11):
        n2 = 13 - n1
        if n2 >= 3:
            configs.append([(n1, 0.5, 0.0), (n2, 1.0, np.pi/n2)])
    
    # Three rings
    for n1 in range(3, 7):
        for n2 in range(3, 13-n1-2):
            n3 = 13 - n1 - n2
            if n3 >= 3:
                configs.append([(n1, 0.3, 0.0), (n2, 0.6, np.pi/n2), (n3, 1.0, np.pi/n3)])
    
    # Center point + ring
    configs.append([(1, 0.0, 0.0), (12, 1.0, 0.0)])
    configs.append([(1, 0.0, 0.0), (11, 1.0, 0.0), (1, 0.0, 0.0)])
    
    return configs

def _neg_symmetric_score(params, config):
    """Negative score for symmetric configuration."""
    pts = _make_symmetric_points(params, config)
    if len(pts) != N:
        return 1e10
    return -_score(pts)

def _neg_symmetric_smooth_score(params, config):
    """Negative smooth score for symmetric configuration."""
    pts = _make_symmetric_points(params, config)
    if len(pts) != N:
        return 1e10
    return -_smooth_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    # Phase 1: Symmetric optimization
    configs = _symmetric_configs()
    rng = np.random.RandomState(42)
    
    for config in configs:
        if time.time() > deadline - 30.0:
            break
        
        n_params = 2 * len(config)
        
        # Try multiple random initializations in parameter space
        for trial in range(5):
            if time.time() > deadline - 25.0:
                break
            
            # Initialize radii and angles
            if trial == 0:
                # Use the base configuration values
                params_init = []
                for n_pts, base_r, base_a in config:
                    params_init.extend([base_r, base_a])
            else:
                # Random perturbation
                params_init = []
                for n_pts, base_r, base_a in config:
                    params_init.extend([
                        base_r * (1 + 0.3 * rng.randn()),
                        base_a + 0.5 * rng.randn()
                    ])
            
            try:
                # Smooth refinement first
                res_smooth = minimize(
                    lambda p: _neg_symmetric_smooth_score(p, config),
                    params_init,
                    method='Powell',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                smooth_params = res_smooth.x
                smooth_pts = _make_symmetric_points(smooth_params, config)
                if len(smooth_pts) == N:
                    smooth_score = _score(smooth_pts)
                    
                    # Exact refinement
                    if time.time() > deadline - 20.0:
                        if smooth_score > best_score:
                            best_score = smooth_score
                            best_pts = smooth_pts.copy()
                        continue
                    
                    res_exact = minimize(
                        lambda p: _neg_symmetric_score(p, config),
                        smooth_params,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    exact_params = res_exact.x
                    exact_pts = _make_symmetric_points(exact_params, config)
                    if len(exact_pts) == N:
                        exact_score = _score(exact_pts)
                        if exact_score > best_score:
                            best_score = exact_score
                            best_pts = exact_pts.copy()
            except Exception:
                pass
    
    # Phase 2: Free-space refinement from best symmetric result
    if time.time() < deadline - 15.0 and best_score > 0:
        try:
            res_free = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            free_pts = res_free.x.reshape(-1, 2)
            free_score = _score(free_pts)
            if free_score > best_score:
                best_score = free_score
                best_pts = free_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Differential evolution if time permits
    if time.time() < deadline - 10.0:
        try:
            bounds = [(-1, 1)] * (2 * N)
            res_de = differential_evolution(
                _neg_score,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                init='latinhypercube'
            )
            de_pts = res_de.x.reshape(-1, 2)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
