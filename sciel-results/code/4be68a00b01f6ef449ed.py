"""Heilbronn-11: maximize minimum triangle area using barycentric coords and gradient ascent."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
S3H = S3 / 2.0
TRI_AREA = S3 / 4.0  # area of unit equilateral triangle

# Deadline: 140s safety margin from 150s limit
DEADLINE = 140

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)  # 165


def _bary_to_cart(bary):
    """Convert barycentric coordinates (n,3) to Cartesian (n,2)."""
    # bary columns: weight for V0, V1, V2
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * S3H
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """Convert Cartesian (n,2) to barycentric (n,3) coordinates."""
    x = pts[:, 0]
    y = pts[:, 1]
    # V0=(0,0), V1=(1,0), V2=(0.5, S3H)
    # b2 = y / S3H
    b2 = y / S3H
    # x = b1 + 0.5*b2 => b1 = x - 0.5*b2
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _project_bary(bary):
    """Project barycentric coords into valid simplex (sum=1, all >= 0)."""
    bary = np.maximum(bary, 0.0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-15)
    return bary / s


def _min_area_from_bary(bary):
    """Compute minimum triangle area from barycentric coordinates."""
    pts = _bary_to_cart(bary)
    return _min_area_cart(pts)


def _min_area_cart(pts):
    """Compute minimum triangle area from Cartesian points (n, 2)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _project_to_triangle_cart(pts):
    """Project Cartesian points into the unit equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _nelder_mead_refine(pts, start_time, deadline, max_iter=500):
    """
    Refine points using Nelder-Mead on the negative minimum area.
    """
    def objective(flat_pts):
        p = flat_pts.reshape(N, 2)
        # Penalty for points outside triangle (should be rare if initialized well)
        # We rely on projection in the wrapper or initial constraint
        # For NM, we just evaluate the raw area.
        return -_min_area_cart(p)
    
    x0 = pts.flatten()
    
    # Nelder-Mead parameters
    options = {
        'maxiter': max_iter,
        'xatol': 1e-6,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options, 
                       callback=lambda x: None)
        
        # Check time
        if (time.time() - start_time) > deadline:
             return pts, _min_area_cart(pts)
             
        best_pts = res.x.reshape(N, 2)
        best_pts = _project_to_triangle_cart(best_pts)
        return best_pts, _min_area_cart(best_pts)
    except Exception:
        return pts, _min_area_cart(pts)


def _coordinate_refine(pts, start_time, deadline, step_init=0.01):
    """
    Coordinate-wise local search (hill climbing).
    """
    current_pts = pts.copy()
    current_val = _min_area_cart(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    step = step_init
    no_improve_count = 0
    
    while (time.time() - start_time) < deadline:
        improved = False
        for i in range(N):
            for d in range(2):
                # Try small steps in positive and negative direction
                for sign in [-1, 1]:
                    cand = current_pts.copy()
                    cand[i, d] += sign * step
                    
                    # Project to triangle
                    cand = _project_to_triangle_cart(cand)
                    
                    val = _min_area_cart(cand)
                    if val > current_val:
                        current_pts = cand
                        current_val = val
                        improved = True
                        if current_val > best_val:
                            best_val = current_val
                            best_pts = current_pts.copy()
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 5:
                break
            step *= 0.8
            
        # Adaptive step size reduction
        step *= 0.95
        
    return best_pts, best_val


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_area_cart(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    # Generate diverse initial configurations
    inits = []
    
    # Init 1: Symmetric distribution (corners, midpoints, center)
    init1_cart = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [0.5, S3H / 3],
        [0.333, S3H / 6],
        [0.667, S3H / 6],
        [0.5, 2 * S3H / 3],
        [0.2, 0.2],
    ], dtype=float)
    inits.append(init1_cart)
    
    # Init 2: Grid-like
    init2_cart = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, S3H / 2],
        [0.75, S3H / 2],
        [0.5, S3H / 2],
        [0.5, S3H / 6],
        [0.5, 2 * S3H / 3],
        [0.5, 0.0],
    ], dtype=float)
    inits.append(init2_cart)
    
    # Init 3: Random spread
    u = rng.random((N, 2))
    mask = u[:, 0] + u[:, 1] > 1
    u[mask, 0] = 1 - u[mask, 0]
    u[mask, 1] = 1 - u[mask, 1]
    x = u[:, 0] + 0.5 * u[:, 1]
    y = u[:, 1] * S3H
    init3_cart = np.column_stack([x, y])
    inits.append(init3_cart)
    
    # Evaluate all inits
    for init in inits:
        init = _project_to_triangle_cart(init)
        _update_best(init)
    
    # Phase A: Nelder-Mead from best init
    if _elapsed_ok(30):
        nm_deadline = start_time + 40
        nm_best, nm_area = _nelder_mead_refine(best_pts, start_time, nm_deadline, max_iter=1000)
        _update_best(nm_best)
    
    # Phase B: Coordinate Refinement
    if _elapsed_ok(10):
        coord_deadline = start_time + 80
        coord_best, coord_area = _coordinate_refine(best_pts, start_time, coord_deadline, step_init=0.05)
        _update_best(coord_best)
        
    # Phase C: More Nelder-Mead from refined result (Polish)
    if _elapsed_ok(10):
        nm_deadline = start_time + 120
        nm_best2, nm_area2 = _nelder_mead_refine(best_pts, start_time, nm_deadline, max_iter=2000)
        _update_best(nm_best2)
        
    # Phase D: Final Coordinate Polish
    if _elapsed_ok(2):
        coord_deadline = start_time + 138
        coord_best2, coord_area2 = _coordinate_refine(best_pts, start_time, coord_deadline, step_init=0.01)
        _update_best(coord_best2)
    
    if best_pts is None:
        # Fallback
        return inits[0]
        
    return _project_to_triangle_cart(best_pts)
