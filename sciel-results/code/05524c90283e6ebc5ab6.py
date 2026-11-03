"""Heilbronn-11 solver: Symmetry-constrained barycentric optimization with soft-min annealing."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 135.0

_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, S3H])

def _bary_to_cart(bary):
    """bary: (N, 3) -> cart: (N, 2)"""
    return bary[:, 0:1] * _V0 + bary[:, 1:2] * _V1 + bary[:, 2:3] * _V2

def _cart_to_bary(cart):
    """cart: (N, 2) -> bary: (N, 3)"""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _min_area_cart(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.min(np.abs(cross) * 0.5)

def _soft_min_area_cart(pts, beta):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    min_a = np.min(areas)
    return min_a - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (areas - min_a))))

def _project_bary(bary):
    """Project barycentric coords to simplex: b>=0, sum(b)=1."""
    b = bary.copy()
    b = np.maximum(0.0, b)
    s = b.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-15, 1.0, s)
    b = b / s
    return b

def _is_valid_cart(pts, tol=1e-9):
    x = pts[:, 0]
    y = pts[:, 1]
    return (np.all(y >= -tol) and 
            np.all(y <= S3H + tol) and 
            np.all(x >= y / S3 - tol) and 
            np.all(x <= 1 - y / S3 + tol))

def _project_cart(pts):
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area = _min_area_cart(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Symmetry structure for 11 points:
    # 3 corners: (0,0), (1,0), (0.5, S3H)
    # 3 mid-edges: (0.5, 0), (0.75, S3H/2), (0.25, S3H/2)
    # 3 on medians at 1/3 from vertex: (1/3, S3H/3), (2/3, S3H/3), (0.5, 2*S3H/3)
    # 1 centroid: (0.5, S3H/3)
    # 1 free interior point
    # This uses 3-fold rotational symmetry for 9 points, leaving 2 free points (centroid + 1 free)
    # Actually, let's use a simpler symmetry: 3-fold rotation about centroid.
    # Under 120-degree rotation, points come in orbits of size 1 (centroid) or 3.
    # 11 = 1 + 3*3 + 2? No. 11 = 1 + 3*3 + 2 doesn't work cleanly.
    # Let's just use the known good symmetric configuration as a strong starting point
    # and optimize all 11 points freely, but with multiple symmetric restarts.
    
    # Known good symmetric starting configuration
    init_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [2/3, S3H/3],
        [0.5, 2*S3H/3],
        [0.5, S3H/3],
        [0.5, S3H/6],
    ], dtype=float)
    
    init_pts = _project_cart(init_pts)
    _update_best(init_pts)
    
    init_bary = _cart_to_bary(init_pts)
    
    def _bary_objective(flat_bary, beta):
        bary = flat_bary.reshape(N, 3)
        bary = _project_bary(bary)
        cart = _bary_to_cart(bary)
        cart = _project_cart(cart)
        ma = _soft_min_area_cart(cart, beta)
        return -ma
    
    def _optimize_bary(bary_start, beta_schedule):
        bary_curr = bary_start.copy()
        for beta in beta_schedule:
            if not _elapsed_ok(10):
                break
            try:
                # Optimize in flattened barycentric space with simplex projection
                result = minimize(
                    lambda x: _bary_objective(x, beta),
                    bary_curr.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                bary_curr = _project_bary(result.x.reshape(N, 3))
            except Exception:
                break
        return bary_curr
    
    beta_schedule = [20.0, 50.0, 150.0, 400.0, 1000.0]
    
    # Multi-start: symmetric perturbations
    n_starts = 30
    for s in range(n_starts):
        if not _elapsed_ok(15):
            break
        
        if s == 0:
            bary_start = init_bary.copy()
        elif s < 10:
            # Small symmetric perturbation
            bary_start = init_bary + rng.normal(0, 0.05, size=(N, 3))
            bary_start = _project_bary(bary_start)
        elif s < 20:
            # Medium perturbation
            bary_start = init_bary + rng.normal(0, 0.15, size=(N, 3))
            bary_start = _project_bary(bary_start)
        else:
            # Larger perturbation
            bary_start = init_bary + rng.normal(0, 0.3, size=(N, 3))
            bary_start = _project_bary(bary_start)
        
        bary_opt = _optimize_bary(bary_start, beta_schedule)
        pts_opt = _bary_to_cart(bary_opt)
        pts_opt = _project_cart(pts_opt)
        _update_best(pts_opt)
    
    # Phase 2: Final polish with hard min via Nelder-Mead in Cartesian
    if best_pts is not None and _elapsed_ok(10):
        def _hard_obj(x):
            pts = x.reshape(N, 2)
            pts = _project_cart(pts)
            return -_min_area_cart(pts)
        
        try:
            result = minimize(
                _hard_obj,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            polished = _project_cart(result.x.reshape(N, 2))
            _update_best(polished)
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = init_pts
        best_area = _min_area_cart(best_pts)
    else:
        best_pts = _project_cart(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
