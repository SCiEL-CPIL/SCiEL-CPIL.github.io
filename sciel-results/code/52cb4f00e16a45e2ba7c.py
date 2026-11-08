"""Heilbronn triangles: maximize min triangle area for 11 points in equilateral triangle."""
N = 11
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

# Precompute triangle indices
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

H = np.sqrt(3) / 2.0
AREA = np.sqrt(3) / 4.0

def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle."""
    h = np.sqrt(3) / 2.0
    for i in range(len(pts)):
        x, y = pts[i]
        # Constraint: y >= 0
        if y < 0:
            y = 0.0
        # Constraint: y <= h * x (left edge)
        if y > h * x:
            y = h * x
        # Constraint: y <= h * (1 - x) (right edge)
        if y > h * (1.0 - x):
            y = h * (1.0 - x)
        pts[i] = [x, y]
    return pts

def _objective_flat(flat_pts):
    """Negative min triangle area for minimization."""
    pts = flat_pts.reshape(N, 2)
    pts = _project_to_triangle(pts.copy())
    return -_min_triangle_area(pts)

def _nelder_mead_step(pts, max_iter=500, timeout=5.0):
    """Run Nelder-Mead optimization on the points."""
    start_time = time.time()
    x0 = pts.flatten()
    
    options = {
        'maxiter': max_iter,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    try:
        res = minimize(
            _objective_flat,
            x0,
            method='Nelder-Mead',
            options=options
        )
        
        if time.time() - start_time > timeout:
            return pts
        
        best_params = res.x
        best_points = best_params.reshape(-1, 2)
        best_points = _project_to_triangle(best_points.copy())
        return best_points
    except Exception:
        return pts

def _slsqp_step(pts, max_iter=200, timeout=5.0):
    """Run SLSQP optimization on the points."""
    start_time = time.time()
    flat_pts = pts.flatten()
    
    try:
        result = minimize(
            _objective_flat,
            flat_pts,
            method='SLSQP',
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        
        if time.time() - start_time > timeout:
            return pts
        
        if result.success or result.fun < _objective_flat(flat_pts):
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project_to_triangle(opt_pts.copy())
            return opt_pts
        return pts
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    h = np.sqrt(3) / 2.0
    
    # Known good starting configuration (symmetric arrangement)
    # Based on literature for 11 points in equilateral triangle
    base_config = np.array([
        [0.5, 0.0],        # bottom center
        [0.0, 0.0],        # bottom left
        [1.0, 0.0],        # bottom right
        [0.25, h * 0.25],  # left mid
        [0.75, h * 0.25],  # right mid
        [0.5, h * 0.5],    # center
        [0.125, h * 0.125],# lower left
        [0.875, h * 0.125],# lower right
        [0.375, h * 0.375],# mid left
        [0.625, h * 0.375],# mid right
        [0.5, h * 0.875],  # top
    ])
    
    best_points = base_config.copy()
    best_area = _min_triangle_area(best_points)
    
    rng = np.random.RandomState(42)
    
    # Phase 1: SLSQP refinement from base and perturbations
    n_attempts = 15
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Start from base config
            pts = base_config.copy()
        else:
            # Random perturbation of best found
            scale = 0.05 if attempt < 5 else 0.02
            pts = best_points + rng.normal(0, scale, size=(N, 2))
            pts = _project_to_triangle(pts.copy())
        
        # Local optimization using SLSQP
        opt_pts = _slsqp_step(pts)
        opt_area = _min_triangle_area(opt_pts)
        
        if opt_area > best_area:
            best_area = opt_area
            best_points = opt_pts.copy()
    
    # Phase 2: Nelder-Mead refinement from current best
    if time.time() < deadline - 10.0:
        nm_pts = _nelder_mead_step(best_points, max_iter=300, timeout=30.0)
        nm_area = _min_triangle_area(nm_pts)
        
        if nm_area > best_area:
            best_area = nm_area
            best_points = nm_pts.copy()
    
    # Phase 3: More SLSQP refinement with tighter tolerance
    if time.time() < deadline - 5.0:
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            opt_pts = _slsqp_step(best_points, max_iter=100, timeout=5.0)
            opt_area = _min_triangle_area(opt_pts)
            
            if opt_area > best_area:
                best_area = opt_area
                best_points = opt_pts.copy()
            else:
                break
    
    return best_points
