"""Heilbronn triangles: maximize the smallest triangle area for 11 points in an equilateral triangle."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0

# Precompute all C(11,3)=165 triangle index triples
_combos = list(combinations(range(N), 3))
_ci = np.array([c[0] for c in _combos])
_cj = np.array([c[1] for c in _combos])
_ck = np.array([c[2] for c in _combos])


def _min_area(pts):
    """Minimum area of any triangle formed by 3 of the 11 points."""
    pi = pts[_ci]
    pj = pts[_cj]
    pk = pts[_ck]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return float(np.min(areas))


def _inside_triangle(pts, tol=1e-9):
    """Check all points are inside (or on boundary of) the equilateral triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    if np.any(y < -tol):
        return False
    if np.any(y > H * x + tol):
        return False
    if np.any(y > H * (1.0 - x) + tol):
        return False
    if np.any(x < -tol) or np.any(x > 1.0 + tol):
        return False
    return True


def _project(pts):
    """Project points onto the triangle (clamp to feasible region)."""
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    x = np.clip(x, 0.0, 1.0)
    y = np.maximum(y, 0.0)
    upper = np.minimum(H * x, H * (1.0 - x))
    y = np.minimum(y, upper)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts


def _neg_min_area(pts):
    """Objective for minimization: negative of minimum triangle area."""
    pts = _project(pts)
    return -_min_area(pts)


def _initial_config_symmetric():
    """Symmetric arrangement based on literature for 11 points in equilateral triangle."""
    base_config = np.array([
        [0.5, 0.0],        # bottom center
        [0.0, 0.0],        # bottom left
        [1.0, 0.0],        # bottom right
        [0.25, H * 0.25],  # left mid
        [0.75, H * 0.25],  # right mid
        [0.5, H * 0.5],    # center
        [0.125, H * 0.125],# lower left
        [0.875, H * 0.125],# lower right
        [0.375, H * 0.375],# mid left
        [0.625, H * 0.375],# mid right
        [0.5, H * 0.875],  # top
    ])
    return _project(base_config)


def _initial_config_grid():
    """Structured grid arrangement within the triangle."""
    pts = []
    # Layer 0 (bottom): 3 points along the base
    for i in range(3):
        pts.append([0.15 + 0.35 * i, 0.05])
    # Layer 1: 4 points
    for i in range(4):
        pts.append([0.1 + 0.25 * i, 0.22])
    # Layer 2: 3 points
    for i in range(3):
        pts.append([0.2 + 0.3 * i, 0.42])
    # Layer 3 (top): 1 point
    pts.append([0.5, 0.55])
    pts = np.array(pts)
    return _project(pts)


def _local_optimize_slsqp(pts, max_time=60.0):
    """Run local optimization using SLSQP starting from pts."""
    best_pts = pts.copy()
    best_score = _min_area(pts)
    t0 = time.time()
    
    def obj(flat_pts):
        p = flat_pts.reshape(N, 2)
        return _neg_min_area(p)
    
    flat_pts = pts.flatten()
    
    try:
        result = minimize(
            obj, flat_pts, method='SLSQP',
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        
        if time.time() - t0 < max_time:
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project(opt_pts)
            opt_score = _min_area(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_score


def _local_optimize_l_bfgsb(pts, max_time=60.0):
    """Run local optimization using L-BFGS-B starting from pts."""
    best_pts = pts.copy()
    best_score = _min_area(pts)
    t0 = time.time()
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def obj(flat_pts):
        p = flat_pts.reshape(N, 2)
        return _neg_min_area(p)
    
    flat_pts = pts.flatten()
    
    try:
        result = minimize(
            obj, flat_pts, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'maxfun': 3000}
        )
        
        if time.time() - t0 < max_time:
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project(opt_pts)
            opt_score = _min_area(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_score


def _local_optimize_nelder_mead(pts, max_time=60.0):
    """Run local optimization using Nelder-Mead starting from pts."""
    best_pts = pts.copy()
    best_score = _min_area(pts)
    t0 = time.time()
    
    def obj(flat_pts):
        p = flat_pts.reshape(N, 2)
        return _neg_min_area(p)
    
    flat_pts = pts.flatten()
    
    try:
        result = minimize(
            obj, flat_pts, method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if time.time() - t0 < max_time:
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _project(opt_pts)
            opt_score = _min_area(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_score


def _multi_start_optimize(pts, max_time=120.0):
    """Multi-start local optimization with multiple optimizers."""
    best_pts = pts.copy()
    best_score = _min_area(pts)
    t_start = time.time()
    
    rng = np.random.RandomState(42)
    
    # Try different optimizers from the initial point
    optimizers = [
        ('SLSQP', _local_optimize_slsqp),
        ('L-BFGS-B', _local_optimize_l_bfgsb),
        ('Nelder-Mead', _local_optimize_nelder_mead),
    ]
    
    for opt_name, opt_fn in optimizers:
        if time.time() - t_start > max_time - 10:
            break
        
        remaining = max_time - (time.time() - t_start)
        if remaining < 5:
            break
        
        try:
            opt_pts, opt_score = opt_fn(best_pts, max_time=min(remaining, 40.0))
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Try random perturbations and re-optimize
    for trial in range(30):
        if time.time() - t_start > max_time - 5:
            break
        
        remaining = max_time - (time.time() - t_start)
        if remaining < 3:
            break
        
        # Random perturbation
        scale = 0.03 * (1.0 - trial / 30.0)  # Decreasing perturbation
        perturbed = best_pts + rng.randn(N, 2) * scale
        perturbed = _project(perturbed)
        
        # Quick local optimization
        try:
            opt_pts, opt_score = _local_optimize_slsqp(perturbed, max_time=min(remaining, 10.0))
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Generate multiple initial configurations
    candidates = []
    
    # Symmetric configuration
    try:
        pts = _initial_config_symmetric()
        if _inside_triangle(pts):
            candidates.append(pts)
    except Exception:
        pass
    
    # Grid configuration
    try:
        pts = _initial_config_grid()
        if _inside_triangle(pts):
            candidates.append(pts)
    except Exception:
        pass
    
    # Random configurations
    rng = np.random.RandomState(123)
    for _ in range(5):
        pts = rng.rand(N, 2)
        pts[:, 0] *= 1.0
        pts[:, 1] *= H
        pts = _project(pts)
        if _inside_triangle(pts):
            candidates.append(pts)
    
    if not candidates:
        # Fallback: simple grid
        pts = np.array([[0.5 * (i % 3), H * 0.1 * (i // 3)] for i in range(N)])
        return _project(pts)
    
    # Evaluate and pick best initial
    best_pts = candidates[0]
    best_score = _min_area(best_pts)
    for pts in candidates[1:]:
        s = _min_area(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Multi-start optimization
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        best_pts, best_score = _multi_start_optimize(best_pts, max_time=min(remaining_time, 120.0))
    
    # Final validation
    best_pts = _project(best_pts)
    if not _inside_triangle(best_pts):
        # Last resort fallback
        best_pts = _initial_config_symmetric()
    
    return best_pts
