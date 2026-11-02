"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: Barycentric coordinates + Soft-Min Annealing + Nelder-Mead Polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
import itertools
import math

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

DEADLINE = 148  # Safety margin from 150s limit

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
    # Inverse of the linear transformation
    # x = 0.5*v + (1-u-v)*0.5 = 0.5 - 0.5*u
    # y = v*S3H + (1-u-v)*S3H = S3H - u*S3H
    # From y: u = (S3H - y) / S3H
    # From x: v = 1 - 2*x
    u = (S3H - pts[:, 1]) / S3H
    v = 1.0 - 2.0 * pts[:, 0]
    return np.column_stack([u, v])

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
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

def _soft_min_area_cart(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip to avoid numerical issues
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _soft_min_gradient(pts, beta):
    """Compute gradient of soft-min area w.r.t. Cartesian points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    shifted = np.clip(shifted, -700, 700)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted))
    soft_min = max_a - log_sum / beta
    
    # Gradient: d(soft_min)/d(point) 
    # soft_min = max_a - (1/beta) * log(sum(exp(-beta*(areas - max_a))))
    # d(soft_min)/d(area_i) = -exp_shifted[i] / sum(exp_shifted)
    weights = -exp_shifted / np.sum(exp_shifted)
    
    # For each triplet, compute contribution to gradient of each point
    # area = 0.5 * |cross| where cross = (pj-pi) x (pk-pi)
    # d(area)/d(pi) = 0.5 * (-(pj_y-pi_y)*(pk_x-pi_x) + (pj_x-pi_x)*(pk_y-pi_y)) ... this gets complex
    # Instead, we'll use finite differences or just return the soft_min value
    # For L-BFGS-B, we can let it use numerical gradients
    
    return soft_min

def _project_bary_to_valid(bary):
    """Ensure barycentric coordinates are within the triangle."""
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    # Enforce u + v <= 1
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        # Scale down u and v to fit
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])

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

    # Initial configurations
    init_bary_list = []
    
    # Base symmetric configuration
    base_sym = np.array([
        [1.0, 0.0],   # V0
        [0.0, 1.0],   # V1
        [0.0, 0.0],   # V2
        [0.5, 0.5],   # Mid V0-V1
        [0.0, 0.5],   # Mid V1-V2
        [0.5, 0.0],   # Mid V2-V0
        [1/3, 1/3],   # Center
        [0.2, 0.2],   # Extra 1
        [0.3, 0.3],   # Extra 2
        [0.4, 0.2],   # Extra 3
        [0.2, 0.4],   # Extra 4
    ])
    init_bary_list.append(base_sym)

    # Generate random initial configurations
    for _ in range(40):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        if np.any(mask):
            scale = 1.0 / (u[mask] + v[mask])
            u[mask] *= scale
            v[mask] *= scale
        init_bary_list.append(np.column_stack([u, v]))
    
    # Add some "near-boundary" configurations
    for _ in range(15):
        u = rng.random(N)
        v = rng.random(N)
        for i in range(N):
            if rng.random() < 0.3:
                edge = rng.integers(0, 3)
                if edge == 0:
                    u[i] = 0
                elif edge == 1:
                    v[i] = 0
                else:
                    s = u[i] + v[i]
                    if s > 0:
                        u[i] /= s
                        v[i] /= s
                    else:
                        u[i] = 0.5
                        v[i] = 0.5
        init_bary_list.append(np.column_stack([u, v]))

    # Optimization Loop
    max_starts = len(init_bary_list)
    current_start = 0
    
    for bary0 in init_bary_list:
        if not _elapsed_ok(25):
            break
        if current_start >= max_starts:
            break
        current_start += 1
        
        # Annealing schedule for beta - more gradual
        betas = [5.0, 15.0, 50.0, 150.0, 500.0, 1500.0, 5000.0]
        
        current_bary = bary0.copy()
        
        for beta in betas:
            if not _elapsed_ok(15):
                break
            
            def objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_soft_min_area_cart(cart, beta)
            
            x0 = current_bary.flatten()
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
                current_bary = res.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # Polish with Nelder-Mead on the hard minimum (exact min)
        if _elapsed_ok(8):
            nm_start = current_bary.flatten()
            
            def nm_objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_min_triangle_area_cart(cart)
            
            try:
                res_nm = minimize(nm_objective, nm_start, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11,
                                           'adaptive': True})
                current_bary = res_nm.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # Random perturbations and re-optimize
        if _elapsed_ok(12) and current_start < max_starts // 2:
            for _ in range(3):
                if not _elapsed_ok(8):
                    break
                perturbed = current_bary + rng.normal(0, 0.04, size=(N, 2))
                perturbed = _project_bary_to_valid(perturbed)
                
                def quick_obj(x):
                    b = x.reshape(N, 2)
                    b = _project_bary_to_valid(b)
                    cart = _bary_to_cart(b)
                    return -_soft_min_area_cart(cart, 1000.0)
                
                try:
                    res_q = minimize(quick_obj, perturbed.flatten(), method='L-BFGS-B',
                                     options={'maxiter': 80, 'ftol': 1e-14})
                    new_bary = res_q.x.reshape(N, 2)
                    new_bary = _project_bary_to_valid(new_bary)
                    _update_best(new_bary)
                    if new_bary is not None:
                        current_bary = new_bary
                except Exception:
                    pass

    # Final polish on the absolute best found
    if best_bary is not None and _elapsed_ok(10):
        def final_nm_obj(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            cart = _bary_to_cart(b)
            return -_min_triangle_area_cart(cart)
        
        try:
            res_final = minimize(final_nm_obj, best_bary.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-13,
                                          'adaptive': True})
            final_bary = res_final.x.reshape(N, 2)
            final_bary = _project_bary_to_valid(final_bary)
            _update_best(final_bary)
        except Exception:
            pass
        
        # Second polish with even tighter tolerance if time permits
        if _elapsed_ok(5):
            try:
                res_final2 = minimize(final_nm_obj, best_bary.flatten(), method='Nelder-Mead',
                                      options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-15,
                                               'adaptive': True})
                final_bary2 = res_final2.x.reshape(N, 2)
                final_bary2 = _project_bary_to_valid(final_bary2)
                _update_best(final_bary2)
            except Exception:
                pass

    if best_cart is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        bary = np.column_stack([u, v])
        best_cart = _bary_to_cart(bary)
        
    return best_cart

# EVOLVE-BLOCK-END
