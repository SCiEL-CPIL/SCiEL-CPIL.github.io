import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

N = 13
# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices. Returns array of shape (_NUM_TRIPLES,)."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _soft_min_objective(x_flat, beta=50.0):
    """
    Smooth surrogate for -min(triangle_area)/hull_area.
    Uses log-sum-exp to approximate min smoothly.
    """
    points = x_flat.reshape(N, 2)
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    
    # Soft min: approx min(a_i) = -1/beta * log(sum(exp(-beta * a_i)))
    # For stability, subtract max first
    a_min_val = np.min(areas)
    if a_min_val < 1e-15:
        return 1e10
    
    # Normalize areas by hull area to get ratio
    ratios = areas / hull_area
    
    # Soft min of ratios
    r_min = np.min(ratios)
    if r_min < 1e-15:
        return 1e10
    
    # Log-sum-exp for soft min: -softmin = 1/beta * log(sum(exp(-beta * r_i)))
    # For numerical stability: shift by r_min
    shifted = -beta * (ratios - r_min)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = r_min + log_sum / beta
    
    return -soft_min_val


def _hard_objective(x_flat):
    """Hard objective: -min_triangle_area / hull_area"""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _perturb(points, rng, sigma=0.05):
    """Add Gaussian noise to points."""
    return points + rng.normal(0, sigma, size=points.shape)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin against 120s limit
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Phase 1: Generate diverse initial configurations
    initial_configs = []
    
    # Regular 13-gon (strong baseline)
    initial_configs.append(_regular_polygon(N, 1.0))
    
    # Perturbed regular polygons
    rng0 = np.random.default_rng(42)
    for sigma in [0.05, 0.1, 0.15, 0.2]:
        initial_configs.append(_perturb(_regular_polygon(N, 1.0), rng0, sigma))
    
    # Rotated regular polygon
    angle_shift = np.pi / N
    angles = np.linspace(angle_shift, 2 * np.pi + angle_shift, N, endpoint=False)
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Two-ring configuration: 7 outer + 6 inner
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    initial_configs.append(np.vstack([outer_pts, inner_pts]))
    
    # Three-ring: 5 outer, 4 middle, 4 inner
    o_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    m_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    i_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    o_pts = np.column_stack([np.cos(o_angles), np.sin(o_angles)])
    m_pts = np.column_stack([0.6 * np.cos(m_angles), 0.6 * np.sin(m_angles)])
    i_pts = np.column_stack([0.3 * np.cos(i_angles), 0.3 * np.sin(i_angles)])
    initial_configs.append(np.vstack([o_pts, m_pts, i_pts]))
    
    # Random configurations
    for seed in range(8):
        rng = np.random.default_rng(seed)
        pts = rng.random((N, 2)) * 2 - 1  # [-1, 1] square
        initial_configs.append(pts)
    
    # Evaluate all initial configs
    for config in initial_configs:
        _update_best(config)
    
    # Phase 2: Coarse optimization with soft-min objective (differentiable)
    # Use L-BFGS-B on the smooth surrogate
    for config in initial_configs:
        if time.time() > deadline - 20:
            break
        
        x0 = config.flatten()
        try:
            # Coarse pass: soft-min with moderate beta
            result = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                args=(30.0,),
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            opt_points = result.x.reshape(N, 2)
            _update_best(opt_points)
        except Exception:
            pass
    
    # Phase 3: Fine polishing with hard objective from best candidates
    # Take top few configurations and polish with Nelder-Mead on hard objective
    if best_points is not None and time.time() < deadline - 10:
        # Polish the best point with Nelder-Mead
        x0 = best_points.flatten()
        try:
            result = minimize(
                _hard_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            opt_points = result.x.reshape(N, 2)
            _update_best(opt_points)
        except Exception:
            pass
    
    # Phase 4: More random restarts if time permits
    while time.time() < deadline - 8:
        rng = np.random.default_rng()
        # Mix of regular polygon perturbations and random
        if rng.random() < 0.5:
            pts = _perturb(_regular_polygon(N, 1.0), rng, sigma=rng.uniform(0.1, 0.3))
        else:
            pts = rng.random((N, 2)) * 2 - 1
        
        x0 = pts.flatten()
        try:
            # Quick soft-min optimization
            result = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                args=(50.0,),
                options={'maxiter': 150, 'ftol': 1e-8}
            )
            opt_points = result.x.reshape(N, 2)
            _update_best(opt_points)
            
            # If improved, do a quick hard polish
            if best_score > 0 and time.time() < deadline - 5:
                try:
                    result2 = minimize(
                        _hard_objective,
                        opt_points.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                    )
                    _update_best(result2.x.reshape(N, 2))
                except Exception:
                    pass
        except Exception:
            pass
    
    # Final check
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    
    return best_points
