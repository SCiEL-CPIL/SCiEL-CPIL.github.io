"""
Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.
Strategy: Barycentric coordinates + Delaunay variance regularization + multi-stage L-BFGS-B + SA polish.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0

def bary_to_cart(bary):
    """Convert barycentric coords (n,3) to cartesian (n,2)."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = H * bary[:, 2]
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert cartesian (n,2) to barycentric coords (n,3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / H
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])

def clamp_bary(bary):
    """Clamp barycentric coordinates to valid simplex."""
    bary = np.maximum(bary, 0.0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-15)
    bary = bary / s
    return bary

def calc_areas_bary(bary):
    """
    Compute areas of all C(11,3) triangles from barycentric coords.
    Area = 0.5 * |det([b_i, b_j, b_k])| * (area of reference triangle)
    Reference triangle area = 0.5 * 1 * H = H/2
    So triangle area = 0.5 * |det| * H/2 = H/4 * |det|
    """
    idx = np.array(list(combinations(range(N), 3)))
    bi = bary[idx[:, 0]]
    bj = bary[idx[:, 1]]
    bk = bary[idx[:, 2]]
    
    # det of 3x3 matrix with rows bi, bj, bk
    det = (bi[:, 0] * (bj[:, 1] * bk[:, 2] - bj[:, 2] * bk[:, 1]) -
           bi[:, 1] * (bj[:, 0] * bk[:, 2] - bj[:, 2] * bk[:, 0]) +
           bi[:, 2] * (bj[:, 0] * bk[:, 1] - bj[:, 1] * bk[:, 0]))
    
    areas = (H / 4.0) * np.abs(det)
    return areas

def _min_triangle_area(cart_pts):
    """Compute minimum triangle area from cartesian points."""
    bary = cart_to_bary(cart_pts)
    areas = calc_areas_bary(bary)
    return np.min(areas)

def _delaunay_variance_term(bary, k=5):
    """
    Compute variance of areas among the k smallest Delaunay triangles.
    This encourages balanced critical triangles.
    Returns 0 if fewer than k triangles.
    """
    cart = bary_to_cart(bary)
    try:
        tri = Delaunay(cart)
    except Exception:
        return 0.0
    
    if tri.simplices is None or len(tri.simplices) == 0:
        return 0.0
    
    # Compute areas of Delaunay triangles
    d_idx = tri.simplices
    if len(d_idx) < 2:
        return 0.0
    
    bi = bary[d_idx[:, 0]]
    bj = bary[d_idx[:, 1]]
    bk = bary[d_idx[:, 2]]
    
    det = (bi[:, 0] * (bj[:, 1] * bk[:, 2] - bj[:, 2] * bk[:, 1]) -
           bi[:, 1] * (bj[:, 0] * bk[:, 2] - bj[:, 2] * bk[:, 0]) +
           bi[:, 2] * (bj[:, 0] * bk[:, 1] - bj[:, 1] * bk[:, 0]))
    
    d_areas = (H / 4.0) * np.abs(det)
    
    # Take the k smallest
    k = min(k, len(d_areas))
    smallest_k = np.sort(d_areas)[:k]
    
    var = np.var(smallest_k)
    return var

def _objective_smooth(params, t=100.0, var_weight=0.1):
    """
    Smooth objective: -soft_min_area + var_weight * delauany_variance + penalty.
    params: barycentric coordinates flattened (n*3,)
    """
    bary = params.reshape(-1, 3)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min
    max_area = np.max(areas)
    if max_area < 1e-15:
        soft_min = 0.0
    else:
        exp_term = np.exp(-t * (areas - max_area))
        log_sum = np.log(np.sum(exp_term))
        soft_min = max_area - (1.0 / t) * log_sum
    
    # Variance term
    var_term = _delaunay_variance_term(bary, k=5)
    
    # Penalty for points outside triangle (should be ~0 with clamping)
    cart = bary_to_cart(bary)
    x, y = cart[:, 0], cart[:, 1]
    d1 = np.maximum(0, -y)
    d2 = np.maximum(0, y - np.sqrt(3) * x)
    d3 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    pen = np.sum(d1**2) + np.sum(d2**2) + np.sum(d3**2)
    
    return -soft_min + var_weight * var_term + 1000.0 * pen

def _objective_hard(params):
    """
    Hard objective for final polish.
    params: barycentric coordinates flattened (n*3,)
    """
    bary = params.reshape(-1, 3)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    area = _min_triangle_area(cart)
    
    x, y = cart[:, 0], cart[:, 1]
    d1 = np.maximum(0, -y)
    d2 = np.maximum(0, y - np.sqrt(3) * x)
    d3 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    pen = np.sum(d1**2) + np.sum(d2**2) + np.sum(d3**2)
    
    return -area + 1000.0 * pen

def generate_hexagonal_init():
    """Generate a hexagonal/lattice-like initial configuration in barycentric coords."""
    # Create a grid in barycentric space
    # Use a triangular lattice pattern
    bary_list = []
    
    # Vertices
    bary_list.append([1.0, 0.0, 0.0])
    bary_list.append([0.0, 1.0, 0.0])
    bary_list.append([0.0, 0.0, 1.0])
    
    # Centroid
    bary_list.append([1/3, 1/3, 1/3])
    
    # Midpoints of edges
    bary_list.append([0.5, 0.5, 0.0])
    bary_list.append([0.5, 0.0, 0.5])
    bary_list.append([0.0, 0.5, 0.5])
    
    # Interior points forming a smaller hexagon
    r = 0.25
    for i in range(6):
        angle = i * np.pi / 3.0
        # In barycentric, a circle around centroid maps to an ellipse
        # Approximate by perturbing centroid
        b0 = 1/3 + r * np.cos(angle) * 0.5
        b1 = 1/3 + r * np.cos(angle + 2*np.pi/3) * 0.5
        b2 = 1/3 + r * np.cos(angle + 4*np.pi/3) * 0.5
        bary_list.append([b0, b1, b2])
    
    bary = np.array(bary_list[:N])
    bary = clamp_bary(bary)
    return bary

def generate_random_init(rng):
    """Generate a random valid configuration in barycentric coords."""
    bary = rng.random((N, 3))
    bary = clamp_bary(bary)
    return bary

def generate_perturbed_init(base_bary, rng, scale=0.1):
    """Generate a perturbed version of a base configuration."""
    bary = base_bary + rng.normal(0, scale, (N, 3))
    bary = clamp_bary(bary)
    return bary

def try_optimize(bary_init, deadline, var_weight=0.1):
    """
    Run multi-stage L-BFGS-B optimization starting from bary_init.
    Returns best cartesian points found.
    """
    best_cart = bary_to_cart(clamp_bary(bary_init))
    best_score = _min_triangle_area(best_cart)
    
    params = bary_init.ravel()
    
    # Stage 1: Low t for basin attraction
    if time.time() < deadline - 10.0:
        res1 = minimize(
            _objective_smooth,
            params,
            args=(30.0, var_weight),
            method='L-BFGS-B',
            options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if res1.fun < _objective_smooth(params, 30.0, var_weight):
            bary1 = clamp_bary(res1.x.reshape(-1, 3))
            cart1 = bary_to_cart(bary1)
            score1 = _min_triangle_area(cart1)
            if score1 > best_score:
                best_score = score1
                best_cart = cart1.copy()
                params = bary1.ravel()
    
    # Stage 2: Medium t
    if time.time() < deadline - 5.0:
        res2 = minimize(
            _objective_smooth,
            params,
            args=(100.0, var_weight),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res2.fun < _objective_smooth(params, 100.0, var_weight):
            bary2 = clamp_bary(res2.x.reshape(-1, 3))
            cart2 = bary_to_cart(bary2)
            score2 = _min_triangle_area(cart2)
            if score2 > best_score:
                best_score = score2
                best_cart = cart2.copy()
                params = bary2.ravel()
    
    # Stage 3: High t, low variance weight
    if time.time() < deadline - 3.0:
        res3 = minimize(
            _objective_smooth,
            params,
            args=(300.0, 0.01),
            method='L-BFGS-B',
            options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res3.fun < _objective_smooth(params, 300.0, 0.01):
            bary3 = clamp_bary(res3.x.reshape(-1, 3))
            cart3 = bary_to_cart(bary3)
            score3 = _min_triangle_area(cart3)
            if score3 > best_score:
                best_score = score3
                best_cart = cart3.copy()
                params = bary3.ravel()
    
    return best_cart, best_score

def nelder_mead_polish(cart_pts, deadline):
    """Polish with Nelder-Mead on hard objective."""
    bary = cart_to_bary(cart_pts)
    params = bary.ravel()
    
    res = minimize(
        _objective_hard,
        params,
        method='Nelder-Mead',
        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    
    if res.fun < _objective_hard(params):
        bary_new = clamp_bary(res.x.reshape(-1, 3))
        cart_new = bary_to_cart(bary_new)
        return cart_new
    return cart_pts

def simulated_annealing(cart_pts, deadline, n_steps=500):
    """
    Simulated annealing polish on the hard objective.
    """
    current = cart_pts.copy()
    current_score = _min_triangle_area(current)
    best = current.copy()
    best_score = current_score
    
    bary_current = cart_to_bary(current)
    
    T = 0.01
    T_min = 0.0001
    alpha = 0.98
    
    rng = np.random.default_rng(123)
    
    for step in range(n_steps):
        if time.time() > deadline - 1.0:
            break
        
        # Perturb in barycentric space
        bary_new = bary_current + rng.normal(0, T * 0.5, (N, 3))
        bary_new = clamp_bary(bary_new)
        cart_new = bary_to_cart(bary_new)
        
        new_score = _min_triangle_area(cart_new)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            bary_current = bary_new
            current = cart_new
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best = cart_new.copy()
        
        T = max(T * alpha, T_min)
    
    return best, best_score

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = []
    
    # Hexagonal lattice init
    inits.append(generate_hexagonal_init())
    
    # Random inits
    for i in range(8):
        inits.append(generate_random_init(rng))
    
    # Perturbed hexagonal inits
    hex_init = generate_hexagonal_init()
    for i in range(4):
        inits.append(generate_perturbed_init(hex_init, rng, scale=0.05 + 0.02 * i))
    
    best_cart = None
    best_score = -1.0
    
    # Multi-start optimization
    for i, bary_init in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        try:
            cart, score = try_optimize(bary_init, deadline, var_weight=0.1)
            
            if score > best_score:
                best_score = score
                best_cart = cart.copy()
                
        except Exception:
            continue
    
    # If we have a good result, polish with Nelder-Mead
    if best_cart is not None and time.time() < deadline - 5.0:
        try:
            cart_nm = nelder_mead_polish(best_cart, deadline)
            score_nm = _min_triangle_area(cart_nm)
            if score_nm > best_score:
                best_score = score_nm
                best_cart = cart_nm.copy()
        except Exception:
            pass
    
    # Simulated annealing polish
    if best_cart is not None and time.time() < deadline - 2.0:
        try:
            cart_sa, score_sa = simulated_annealing(best_cart, deadline, n_steps=300)
            if score_sa > best_score:
                best_score = score_sa
                best_cart = cart_sa.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead if time permits
    if best_cart is not None and time.time() < deadline - 1.0:
        try:
            cart_final = nelder_mead_polish(best_cart, deadline)
            score_final = _min_triangle_area(cart_final)
            if score_final > best_score:
                best_score = score_final
                best_cart = cart_final.copy()
        except Exception:
            pass
    
    # Fallback
    if best_cart is None:
        bary_fallback = generate_hexagonal_init()
        best_cart = bary_to_cart(bary_fallback)
    
    return best_cart
