"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: D3-symmetry-enforced parameterization + multi-restart Nelder-Mead + SA polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
import itertools
import math

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIPLETS = len(_TRIPLETS)

DEADLINE = 145  # Safety margin from 150s limit

# D3 symmetry group for equilateral triangle
# Rotation by 120 degrees around center (1/3, S3H/3)
# Center of equilateral triangle
CENTER = np.array([1.0/3, S3H/3])

def _rotate_120(pt):
    """Rotate point 120 degrees around triangle center."""
    dx = pt[0] - CENTER[0]
    dy = pt[1] - CENTER[1]
    cos_120 = -0.5
    sin_120 = S3H
    rx = cos_120 * dx - sin_120 * dy
    ry = sin_120 * dx + cos_120 * dy
    return np.array([CENTER[0] + rx, CENTER[1] + ry])

def _rotate_240(pt):
    """Rotate point 240 degrees around triangle center."""
    return _rotate_120(_rotate_120(pt))

def _reflect(pt):
    """Reflect point across vertical axis x = 1/2."""
    return np.array([1.0 - pt[0], pt[1]])

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    bary: shape (N, 2) where u, v >= 0 and u+v <= 1.
    P = u*V0 + v*V1 + (1-u-v)*V2
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """
    Convert Cartesian to barycentric (u, v).
    """
    u = (S3H - pts[:, 1]) / S3H
    v = 1.0 - 2.0 * pts[:, 0]
    return np.column_stack([u, v])

def _all_triangle_areas_cart(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points."""
    return np.min(_all_triangle_areas_cart(pts))

def _project_bary_to_valid(bary):
    """Ensure barycentric coordinates are within the triangle."""
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])

def _symmetric_bary_from_params(params):
    """
    Generate 11 barycentric coordinates from a reduced set of parameters
    using D3 symmetry.
    
    params: array of shape (6,)
    - params[0:3]: barycentric coords of one representative point (orbit 1)
    - params[3:6]: barycentric coords of another representative point (orbit 2)
    
    The 11 points are:
    - 3 vertices (fixed): V0, V1, V2
    - 3 midpoints (fixed): midpoints of edges
    - 1 center (fixed)
    - 2 free points with D3 symmetry: each generates 3 points via rotation
    
    Wait, that's 3+3+1+6 = 13. Too many.
    
    Let me reconsider: 11 points total.
    With D3 symmetry:
    - 3 vertices (orbit of size 3 under rotation)
    - 1 center (fixed point)
    - 3 midpoints (orbit of size 3)
    - 4 remaining points: must form orbits of size 1 or 3
    - 4 = 3 + 1 or 1 + 1 + 1 + 1 or 3 + 1
    
    Let's use: 3 vertices + 3 midpoints + 1 center + 3 from one orbit + 1 on axis = 11
    The point on the axis (x=1/2) is fixed under reflection.
    
    Actually, let me just use 5 free parameters with symmetry:
    - 3 vertices fixed
    - 1 center fixed  
    - 3 midpoints fixed
    - 2 free orbits of 3 points each = 6 points
    Total: 3 + 1 + 3 + 6 = 13. Still too many.
    
    OK let me just do: 3 vertices + 1 center + 3 midpoints + 4 free points (no symmetry) = 11
    But that breaks symmetry.
    
    Alternative: use full symmetry with:
    - 3 vertices
    - 1 center
    - 3 midpoints  
    - 3 points from one orbit (rotated copies of one point)
    - 1 point on the vertical axis (x=0.5)
    Total: 3+1+3+3+1 = 11
    
    params: 4 values
    - params[0:2]: barycentric of the orbit representative
    - params[2:4]: barycentric of the axis point
    """
    # Fixed points
    bary = np.zeros((N, 2))
    
    # Vertices: V0=(1,0), V1=(0,1), V2=(0,0) in barycentric
    bary[0] = [1.0, 0.0]  # V0
    bary[1] = [0.0, 1.0]  # V1
    bary[2] = [0.0, 0.0]  # V2
    
    # Midpoints
    bary[3] = [0.5, 0.5]   # mid V0-V1
    bary[4] = [0.0, 0.5]   # mid V1-V2
    bary[5] = [0.5, 0.0]   # mid V2-V0
    
    # Center
    bary[6] = [1/3, 1/3]
    
    # Orbit of 3 points from params[0:2]
    rep_bary = np.array([params[0], params[1]])
    rep_bary = _project_bary_to_valid(rep_bary.reshape(1, 2))[0]
    rep_cart = _bary_to_cart(rep_bary.reshape(1, 2))[0]
    
    bary[7] = rep_bary
    rot1 = _rotate_120(rep_cart)
    rot2 = _rotate_240(rep_cart)
    bary[8] = _cart_to_bary(rot1.reshape(1, 2))[0]
    bary[9] = _cart_to_bary(rot2.reshape(1, 2))[0]
    
    # Axis point from params[2:4]
    axis_bary = np.array([params[2], params[3]])
    axis_bary = _project_bary_to_valid(axis_bary.reshape(1, 2))[0]
    bary[10] = axis_bary
    
    return bary

def _symmetric_objective(params):
    """Objective function for symmetric parameterization."""
    bary = _symmetric_bary_from_params(params)
    cart = _bary_to_cart(bary)
    return -_min_triangle_area_cart(cart)

def _soft_min_area_cart(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas_cart(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _soft_symmetric_objective(params, beta):
    """Soft objective for symmetric parameterization."""
    bary = _symmetric_bary_from_params(params)
    cart = _bary_to_cart(bary)
    return -_soft_min_area_cart(cart, beta)

def _largest_empty_triangle_move(pts, bary):
    """
    Find the minimum-area triangle and move the most central point towards
    the centroid of the other points.
    """
    areas = _all_triangle_areas_cart(pts)
    min_idx = np.argmin(areas)
    min_tri = _TRIPLETS[min_idx]
    
    other_indices = [i for i in range(N) if i not in min_tri]
    other_pts = pts[other_indices]
    
    centroid = np.mean(other_pts, axis=0)
    
    centroid_bary = _cart_to_bary(centroid.reshape(1, 2))[0]
    centroid_bary = np.clip(centroid_bary, 0, 1)
    s = centroid_bary[0] + centroid_bary[1]
    if s > 1:
        centroid_bary /= s
    
    distances = [np.linalg.norm(pts[i] - centroid) for i in min_tri]
    replace_idx = min_tri[np.argmin(distances)]
    
    new_bary = bary.copy()
    new_bary[replace_idx] = centroid_bary
    return new_bary

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0
    best_cart = None

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area, best_cart
        bary = _project_bary_to_valid(bary)
        cart = _bary_to_cart(bary)
        area = _min_triangle_area_cart(cart)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
            best_cart = cart.copy()

    # === Phase 1: Symmetric optimization with multiple restarts ===
    # Use D3 symmetry to reduce search space
    n_sym_restarts = 15
    for r in range(n_sym_restarts):
        if not _elapsed_ok(30):
            break
        
        # Random initial parameters
        if r == 0:
            # Start near known good configuration
            params0 = np.array([1/3, 1/3, 0.5, 0.0])
        else:
            params0 = rng.random(4)
            params0[0] = np.clip(params0[0], 0.05, 0.9)
            params0[1] = np.clip(params0[1], 0.05, 0.9)
            params0[2] = np.clip(params0[2], 0.0, 1.0)
            params0[3] = np.clip(params0[3], 0.0, 1.0)
            # Ensure u+v <= 1 for first point
            if params0[0] + params0[1] > 0.9:
                params0[0] *= 0.5
                params0[1] *= 0.5
        
        # Nelder-Mead on symmetric parameters
        try:
            res = minimize(_symmetric_objective, params0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            params_best = res.x
            bary = _symmetric_bary_from_params(params_best)
            _update_best(bary)
        except Exception:
            pass

    # === Phase 2: Full-space optimization from symmetric solutions ===
    if best_bary is not None and _elapsed_ok(40):
        # Use the best symmetric solution as starting point for full optimization
        full_start = best_bary.copy()
        
        # Nelder-Mead in full space
        def full_obj(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            cart = _bary_to_cart(b)
            return -_min_triangle_area_cart(cart)
        
        try:
            res_full = minimize(full_obj, full_start.flatten(), method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13,
                                        'adaptive': True})
            bary_full = res_full.x.reshape(N, 2)
            _update_best(bary_full)
        except Exception:
            pass

    # === Phase 3: Simulated Annealing ===
    if _elapsed_ok(30):
        current_bary = best_bary.copy() if best_bary is not None else _symmetric_bary_from_params(np.array([1/3, 1/3, 0.5, 0.0]))
        current_bary = _project_bary_to_valid(current_bary)
        
        T_start = 0.03
        T_end = 0.0005
        n_sa_iter = 3000
        
        for it in range(n_sa_iter):
            if not _elapsed_ok(5):
                break
            T = T_start * (T_end / T_start) ** (it / n_sa_iter)
            
            perturbation_scale = 0.04 * (T / T_start) ** 0.5
            delta = rng.normal(0, perturbation_scale, size=(N, 2))
            new_bary = current_bary + delta
            new_bary = _project_bary_to_valid(new_bary)
            
            new_cart = _bary_to_cart(new_bary)
            new_area = _min_triangle_area_cart(new_cart)
            cur_cart = _bary_to_cart(current_bary)
            cur_area = _min_triangle_area_cart(cur_cart)
            
            delta_f = new_area - cur_area
            
            if delta_f > 0 or rng.random() < np.exp(delta_f / T):
                current_bary = new_bary
                _update_best(current_bary)

    # === Phase 4: L-BFGS-B with soft minimum ===
    if best_bary is not None and _elapsed_ok(20):
        for beta in [200.0, 1000.0, 5000.0]:
            if not _elapsed_ok(5):
                break
            
            def soft_obj(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_soft_min_area_cart(cart, beta)
            
            try:
                res_l = minimize(soft_obj, best_bary.flatten(), method='L-BFGS-B',
                                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                bary_l = res_l.x.reshape(N, 2)
                _update_best(bary_l)
            except Exception:
                pass

    # === Phase 5: Final polish ===
    if best_bary is not None and _elapsed_ok(10):
        def final_obj(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            cart = _bary_to_cart(b)
            return -_min_triangle_area_cart(cart)
        
        try:
            res_final = minimize(final_obj, best_bary.flatten(), method='Nelder-Mead',
                                options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-14,
                                         'adaptive': True})
            bary_final = res_final.x.reshape(N, 2)
            _update_best(bary_final)
        except Exception:
            pass

        # One more polish if time allows
        if _elapsed_ok(5):
            try:
                res_final2 = minimize(final_obj, best_bary.flatten(), method='Nelder-Mead',
                                     options={'maxiter': 400, 'xatol': 1e-13, 'fatol': 1e-15,
                                              'adaptive': True})
                bary_final2 = res_final2.x.reshape(N, 2)
                _update_best(bary_final2)
            except Exception:
                pass

    if best_cart is None:
        # Fallback: symmetric configuration
        bary = _symmetric_bary_from_params(np.array([1/3, 1/3, 0.5, 0.0]))
        best_cart = _bary_to_cart(bary)
        
    return best_cart
