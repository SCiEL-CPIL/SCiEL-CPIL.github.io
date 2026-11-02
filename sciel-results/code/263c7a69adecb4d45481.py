"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: Sobol-Grid initialization + Local Nudge (move points in smallest triangles)
         + Soft-Min Annealing + Nelder-Mead Polish + SLSQP constraint-based optimization.
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
_N_TRIP = len(_TRIPLETS)  # 165

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


def _soft_min_area_cart(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas_cart(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


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


def _local_nudge(bary, max_iters=200, step_scale=0.02, rng=None):
    """
    Local search: identify points involved in the smallest triangles and nudge them
    to break near-degeneracies. This is a direct, non-gradient approach.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current_bary = bary.copy()
    current_cart = _bary_to_cart(current_bary)
    current_min = _min_triangle_area_cart(current_cart)
    
    for it in range(max_iters):
        areas = _all_triangle_areas_cart(current_cart)
        min_idx = np.argmin(areas)
        
        # Get the three points in the smallest triangle
        i, j, k = _TRIPLETS[min_idx]
        involved = [i, j, k]
        
        # Try nudging each involved point in random directions
        best_nudge = None
        best_new_min = current_min
        
        for pt_idx in involved:
            # Try several random nudge directions
            for _ in range(8):
                # Random direction in barycentric space
                du = rng.normal(0, step_scale)
                dv = rng.normal(0, step_scale)
                
                new_bary = current_bary.copy()
                new_bary[pt_idx, 0] += du
                new_bary[pt_idx, 1] += dv
                new_bary = _project_bary_to_valid(new_bary)
                
                new_cart = _bary_to_cart(new_bary)
                new_min = _min_triangle_area_cart(new_cart)
                
                if new_min > best_new_min:
                    best_new_min = new_min
                    best_nudge = new_bary.copy()
        
        if best_nudge is not None:
            current_bary = best_nudge
            current_cart = _bary_to_cart(current_bary)
            current_min = best_new_min
        else:
            # No improvement; reduce step size
            step_scale *= 0.7
            if step_scale < 1e-6:
                break
    
    return current_bary, current_min


def _slsqp_optimize(bary, max_iter=50):
    """
    Use SLSQP to maximize the minimum triangle area directly with constraints.
    The objective is to maximize t such that all triangle areas >= t.
    """
    def objective(x):
        b = x.reshape(N, 2)
        b = _project_bary_to_valid(b)
        cart = _bary_to_cart(b)
        return -_min_triangle_area_cart(cart)
    
    # Constraints: points must be inside the triangle
    # In barycentric: u >= 0, v >= 0, u + v <= 1
    def constraint_u(x):
        b = x.reshape(N, 2)
        return b[:, 0]
    
    def constraint_v(x):
        b = x.reshape(N, 2)
        return b[:, 1]
    
    def constraint_sum(x):
        b = x.reshape(N, 2)
        return 1.0 - b[:, 0] - b[:, 1]
    
    cons = [
        {'type': 'ineq', 'fun': constraint_u},
        {'type': 'ineq', 'fun': constraint_v},
        {'type': 'ineq', 'fun': constraint_sum}
    ]
    
    try:
        res = minimize(objective, bary.flatten(), method='SLSQP',
                       constraints=cons, options={'maxiter': max_iter, 'ftol': 1e-12})
        if res.success or res.fun < objective(bary.flatten()):
            new_bary = res.x.reshape(N, 2)
            new_bary = _project_bary_to_valid(new_bary)
            return new_bary
    except Exception:
        pass
    return bary


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

    # Generate initial configurations using Sobol-like quasi-random + structured patterns
    init_bary_list = []
    
    # 1. Symmetric base configuration
    base_sym = np.array([
        [1.0, 0.0],   # V0
        [0.0, 1.0],   # V1
        [0.0, 0.0],   # V2
        [0.5, 0.5],   # Mid V0-V1
        [0.0, 0.5],   # Mid V1-V2
        [0.5, 0.0],   # Mid V2-V0
        [1/3, 1/3],   # Center
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.2],
        [0.2, 0.4],
    ])
    init_bary_list.append(base_sym)
    
    # 2. Quasi-random (Sobol-like) configurations using Halton sequence
    def halton(index, base):
        """Generate the index-th Halton sequence number in the given base."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    for start_idx in range(20):
        bary = np.zeros((N, 2))
        for i in range(N):
            idx = start_idx * N + i + 1
            u = halton(idx, 2)
            v = halton(idx, 3)
            # Map to triangle: ensure u+v <= 1
            s = u + v
            if s > 1.0:
                u /= s
                v /= s
            bary[i] = [u, v]
        init_bary_list.append(bary)
    
    # 3. Random configurations
    for _ in range(30):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        if np.any(mask):
            scale = 1.0 / (u[mask] + v[mask])
            u[mask] *= scale
            v[mask] *= scale
        init_bary_list.append(np.column_stack([u, v]))
    
    # 4. Near-boundary configurations
    for _ in range(10):
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
    
    # Sort initial configs by initial quality (best first)
    init_bary_list.sort(key=lambda b: -_min_triangle_area_cart(_bary_to_cart(_project_bary_to_valid(b))))
    
    # Optimization Loop
    for bary0 in init_bary_list:
        if not _elapsed_ok(30):
            break
        
        current_bary = bary0.copy()
        _update_best(current_bary)
        
        # Phase 1: Soft-min annealing with L-BFGS-B
        betas = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
        
        for beta in betas:
            if not _elapsed_ok(20):
                break
            
            def objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_soft_min_area_cart(cart, beta)
            
            x0 = current_bary.flatten()
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
                current_bary = res.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # Phase 2: Local Nudge (targeted moves on smallest triangles)
        if _elapsed_ok(15):
            current_bary, _ = _local_nudge(current_bary, max_iters=150, step_scale=0.02, rng=rng)
            _update_best(current_bary)
        
        # Phase 3: SLSQP optimization (handles non-smooth min directly)
        if _elapsed_ok(12):
            current_bary = _slsqp_optimize(current_bary, max_iter=30)
            _update_best(current_bary)
        
        # Phase 4: Nelder-Mead polish on hard minimum
        if _elapsed_ok(10):
            def nm_objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_min_triangle_area_cart(cart)
            
            try:
                res_nm = minimize(nm_objective, current_bary.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                           'adaptive': True})
                current_bary = res_nm.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # Phase 5: More local nudges with smaller steps
        if _elapsed_ok(8):
            current_bary, _ = _local_nudge(current_bary, max_iters=100, step_scale=0.005, rng=rng)
            _update_best(current_bary)
        
        # Phase 6: Another SLSQP pass
        if _elapsed_ok(8):
            current_bary = _slsqp_optimize(current_bary, max_iter=20)
            _update_best(current_bary)
        
        # Random perturbations and re-optimize for early starts
        if _elapsed_ok(15) and init_bary_list.index(bary0) < len(init_bary_list) // 3:
            for _ in range(2):
                if not _elapsed_ok(10):
                    break
                perturbed = current_bary + rng.normal(0, 0.03, size=(N, 2))
                perturbed = _project_bary_to_valid(perturbed)
                
                # Quick soft-min optimization
                def quick_obj(x):
                    b = x.reshape(N, 2)
                    b = _project_bary_to_valid(b)
                    cart = _bary_to_cart(b)
                    return -_soft_min_area_cart(cart, 500.0)
                
                try:
                    res_q = minimize(quick_obj, perturbed.flatten(), method='L-BFGS-B',
                                     options={'maxiter': 50, 'ftol': 1e-14})
                    new_bary = res_q.x.reshape(N, 2)
                    new_bary = _project_bary_to_valid(new_bary)
                    _update_best(new_bary)
                    current_bary = new_bary
                except Exception:
                    pass
                
                # Local nudge on the perturbed result
                if _elapsed_ok(5):
                    current_bary, _ = _local_nudge(current_bary, max_iters=50, step_scale=0.01, rng=rng)
                    _update_best(current_bary)

    # Final high-precision polish on the best found
    if best_bary is not None and _elapsed_ok(12):
        def final_nm_obj(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            cart = _bary_to_cart(b)
            return -_min_triangle_area_cart(cart)
        
        try:
            res_final = minimize(final_nm_obj, best_bary.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-14,
                                          'adaptive': True})
            final_bary = res_final.x.reshape(N, 2)
            final_bary = _project_bary_to_valid(final_bary)
            _update_best(final_bary)
        except Exception:
            pass
        
        # SLSQP polish
        if _elapsed_ok(8):
            best_bary = _slsqp_optimize(best_bary, max_iter=40)
            _update_best(best_bary)
        
        # Final local nudge with very small steps
        if _elapsed_ok(5):
            best_bary, _ = _local_nudge(best_bary, max_iters=200, step_scale=0.001, rng=rng)
            _update_best(best_bary)
        
        # One more NM polish
        if _elapsed_ok(5):
            try:
                res_final2 = minimize(final_nm_obj, best_bary.flatten(), method='Nelder-Mead',
                                      options={'maxiter': 400, 'xatol': 1e-14, 'fatol': 1e-16,
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
