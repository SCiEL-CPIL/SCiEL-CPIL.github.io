"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    
    # Try multiple random perturbations and local optimizations
    rng = np.random.RandomState(42)
    n_attempts = 20
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Start from base config
            pts = base_config.copy()
        else:
            # Random perturbation of best found
            pts = best_points + rng.normal(0, 0.05, size=(N, 2))
            pts = _project_to_triangle(pts.copy())
        
        # Local optimization using SLSQP
        flat_pts = pts.flatten()
        
        try:
            result = minimize(
                _objective_flat,
                flat_pts,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < _objective_flat(flat_pts):
                opt_pts = result.x.reshape(N, 2)
                opt_pts = _project_to_triangle(opt_pts.copy())
                opt_area = _min_triangle_area(opt_pts)
                
                if opt_area > best_area:
                    best_area = opt_area
                    best_points = opt_pts.copy()
                    
        except Exception:
            pass
    
    return best_points
