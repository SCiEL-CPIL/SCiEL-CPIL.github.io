"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: D3-symmetry-enforced parameterization + Nelder-Mead on reduced space.
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

# D3 symmetry: 3-fold rotational + 3 reflections
# 11 points = 3 orbits of size 3 + 1 orbit of size 1 + 1 orbit of size 1 + 1 orbit of size 3
# Actually: 11 = 1 (center) + 3 (one orbit) + 3 (one orbit) + 3 (one orbit) + 1 (on axis) 
# Let's use: 1 center + 3 orbits of 3 = 10... need 11.
# 11 = 1 (center) + 3 + 3 + 3 + 1 (on 3-fold axis)
# Or: 3 orbits of 3 + 2 orbits of 1 = 11
# D3 orbits: size 1 (center, or on axes), size 3 (generic), size 6 (if not on reflection axis)
# For equilateral triangle with D3: orbits have size 1 (center), 3 (on 3-fold axis), or 6 (generic)
# 11 = 1 + 3 + 3 + 3 + 1? No, 1+3+3+3+1=11. But two size-1 orbits: center + one on axis.
# Or 11 = 3 + 3 + 3 + 1 + 1 (two axis points + three orbits)
# Let's parameterize: center + 3 points on 3-fold axes + 3 more on 3-fold axes? 
# Simpler: use 3 orbits of 3 + 2 fixed points (center + one on axis)
# Parameters: center (2D -> but fixed at centroid), 
#   orbit A: 1 point (x,y) -> 3 rotated copies
#   orbit B: 1 point (x,y) -> 3 rotated copies  
#   orbit C: 1 point (x,y) -> 3 rotated copies
#   fixed1: center (0 params, known)
#   fixed2: on 3-fold axis (1 param: distance from center along axis)
# Total params: 2+2+2+1 = 7

def _rotate_120(p):
    """Rotate point 120 degrees around centroid of triangle."""
    cx, cy = 0.5, S3H / 3  # centroid of equilateral triangle
    # Centroid is at (0.5, sqrt(3)/6)
    cx = 0.5
    cy = S3H / 3
    dx = p[0] - cx
    dy = p[1] - cy
    cos120 = -0.5
    sin120 = S3 / 2
    rx = cos120 * dx - sin120 * dy + cx
    ry = sin120 * dx + cos120 * dy + cy
    return np.array([rx, ry])

def _rotate_240(p):
    """Rotate point 240 degrees around centroid of triangle."""
    cx = 0.5
    cy = S3H / 3
    dx = p[0] - cx
    dy = p[1] - cy
    cos240 = -0.5
    sin240 = -S3 / 2
    rx = cos240 * dx - sin240 * dy + cx
    ry = sin240 * dx + cos240 * dy + cy
    return np.array([rx, ry])

def _symmetric_points_from_params(params):
    """
    Generate 11 symmetric points from 7 parameters.
    params = [ax, ay, bx, by, cx, cy, d]
    where (ax,ay), (bx,by), (cx,cy) are representatives of 3 orbits,
    and d is distance along the 3-fold axis for the fixed point.
    """
    cx, cy = 0.5, S3H / 3  # centroid
    
    # Orbit A: 3 points
    pa = np.array([params[0], params[1]])
    pa2 = _rotate_120(pa)
    pa3 = _rotate_240(pa)
    
    # Orbit B: 3 points
    pb = np.array([params[2], params[3]])
    pb2 = _rotate_120(pb)
    pb3 = _rotate_240(pb)
    
    # Orbit C: 3 points
    pc = np.array([params[4], params[5]])
    pc2 = _rotate_120(pc)
    pc3 = _rotate_240(pc)
    
    # Fixed point 1: center
    p_center = np.array([cx, cy])
    
    # Fixed point 2: on 3-fold axis (vertical axis through centroid)
    d = params[6]
    p_axis = np.array([cx, cy + d])
    
    pts = np.array([pa, pa2, pa3, pb, pb2, pb3, pc, pc2, pc3, p_center, p_axis])
    return pts

def _project_to_triangle(pts):
    """Project points to be inside the triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        # Check if inside: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
        if y < 0:
            y = 0
        if y > S3H * x:
            # Project to line y = sqrt(3)*x
            if x > 0:
                y = S3H * x
            else:
                y = 0
                x = 0
        if y > S3H * (1 - x):
            if (1 - x) > 0:
                y = S3H * (1 - x)
            else:
                y = 0
                x = 1
        pts[i] = [x, y]
    return pts

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

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_params = None
    best_area = 0.0
    best_cart = None

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _eval_params(params):
        """Evaluate symmetric configuration from 7 parameters."""
        pts = _symmetric_points_from_params(params)
        pts = _project_to_triangle(pts)
        area = _min_triangle_area_cart(pts)
        return area, pts

    def _update_best(params, pts, area):
        nonlocal best_params, best_area, best_cart
        if area > best_area:
            best_area = area
            best_params = params.copy()
            best_cart = pts.copy()

    # Objective for Nelder-Mead in reduced 7D space
    def _symmetric_objective(params):
        pts = _symmetric_points_from_params(params)
        pts = _project_to_triangle(pts)
        return -_min_triangle_area_cart(pts)

    # Generate initial parameter sets
    init_params_list = []
    
    # Known good symmetric starting point
    # Center at centroid, orbits at various distances
    cx, cy = 0.5, S3H / 3
    # Orbit A: near vertices
    r_a = 0.3
    a_angle = 0  # along x-axis from centroid
    pa = np.array([cx + r_a * math.cos(a_angle), cy + r_a * math.sin(a_angle)])
    # Orbit B: medium
    r_b = 0.15
    b_angle = math.pi / 6
    pb = np.array([cx + r_b * math.cos(b_angle), cy + r_b * math.sin(b_angle)])
    # Orbit C: small
    r_c = 0.08
    c_angle = math.pi / 3
    pc = np.array([cx + r_c * math.cos(c_angle), cy + r_c * math.sin(c_angle)])
    d_axis = 0.1  # distance along vertical axis
    
    init_params_list.append(np.array([pa[0], pa[1], pb[0], pb[1], pc[0], pc[1], d_axis]))

    # Random initial parameters
    for _ in range(20):
        pa = np.array([cx + rng.uniform(-0.4, 0.4), cy + rng.uniform(-0.3, 0.3)])
        pb = np.array([cx + rng.uniform(-0.3, 0.3), cy + rng.uniform(-0.25, 0.25)])
        pc = np.array([cx + rng.uniform(-0.2, 0.2), cy + rng.uniform(-0.2, 0.2)])
        d_axis = rng.uniform(-0.3, 0.3)
        init_params_list.append(np.array([pa[0], pa[1], pb[0], pb[1], pc[0], pc[1], d_axis]))

    # Try each initial parameter set
    for p0 in init_params_list:
        if not _elapsed_ok(15):
            break
        
        # Evaluate initial
        area0, pts0 = _eval_params(p0)
        _update_best(p0, pts0, area0)
        
        # Nelder-Mead optimization in 7D
        if _elapsed_ok(10):
            try:
                res = minimize(_symmetric_objective, p0.copy(), method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True})
                area_res, pts_res = _eval_params(res.x)
                _update_best(res.x, pts_res, area_res)
            except Exception:
                pass
        
        # Additional Nelder-Mead from best if improved
        if best_params is not None and _elapsed_ok(8):
            try:
                res2 = minimize(_symmetric_objective, best_params.copy(), method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14,
                                        'adaptive': True})
                area_res2, pts_res2 = _eval_params(res2.x)
                _update_best(res2.x, pts_res2, area_res2)
            except Exception:
                pass

    # If we have a good result, do final polish
    if best_params is not None and _elapsed_ok(10):
        try:
            res_final = minimize(_symmetric_objective, best_params.copy(), method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16,
                                         'adaptive': True})
            area_final, pts_final = _eval_params(res_final.x)
            _update_best(res_final.x, pts_final, area_final)
        except Exception:
            pass

    if best_cart is None:
        # Fallback: random points
        pts = rng.random((N, 2))
        pts = _project_to_triangle(pts)
        best_cart = pts

    return best_cart
