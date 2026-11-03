"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def _all_triples(n):
    """Precompute all triples for n points."""
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    return triples

def _compute_areas_and_grads(pts):
    """
    Compute all triangle areas and their gradients.
    Returns (areas, grad_list) where grad_list[idx] is (g_i, g_j, g_k) for triple idx.
    """
    n = len(pts)
    triples = _all_triples(n)
    areas = np.zeros(len(triples))
    grad_list = []
    
    for idx, (i, j, k) in enumerate(triples):
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        cross_val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1])
        area = 0.5 * abs(cross_val)
        areas[idx] = area
        
        sign_cross = np.sign(cross_val)
        if sign_cross == 0:
            sign_cross = 1.0
        
        g_pi = 0.5 * sign_cross * np.array([pi[1] - pk[1], pi[0] - pj[0]])
        g_pj = 0.5 * sign_cross * np.array([pi[0] - pk[0], pk[1] - pi[1]])
        g_pk = 0.5 * sign_cross * np.array([pi[1] - pj[1], pj[0] - pi[0]])
        
        grad_list.append((g_pi, g_pj, g_pk))
    
    return areas, grad_list

def _sqp_objective_and_grad(pts_flat, active_triples, tol=1e-6):
    """
    SQP-style objective: minimize -min_area where min_area is approximated
    by the average of the smallest few triangle areas (active set).
    Returns (value, gradient) w.r.t. flat point coordinates.
    """
    n = N
    pts = pts_flat.reshape(-1, 2)
    areas, grad_list = _compute_areas_and_grads(pts)
    
    # Find the active set: triangles with area close to the minimum
    min_area = np.min(areas)
    active_mask = areas < (min_area + tol)
    
    if not np.any(active_mask):
        active_mask = np.zeros(len(areas), dtype=bool)
        active_mask[np.argmin(areas)] = True
    
    # Objective: negative of the mean of active areas
    active_areas = areas[active_mask]
    value = -np.mean(active_areas)
    
    # Gradient: average of gradients of active triangles
    grad = np.zeros((n, 2))
    active_indices = np.where(active_mask)[0]
    for idx in active_indices:
        i, j, k = _all_triples(n)[idx]
        g_pi, g_pj, g_pk = grad_list[idx]
        grad[i] += g_pi
        grad[j] += g_j
        grad[k] += g_pk
    
    num_active = len(active_indices)
    if num_active > 0:
        grad /= num_active
    
    return value, -grad.flatten()

def _sqp_objective_value_only(pts_flat, tol=1e-6):
    """Value-only version for Nelder-Mead."""
    n = N
    pts = pts_flat.reshape(-1, 2)
    areas, _ = _compute_areas_and_grads(pts)
    min_area = np.min(areas)
    active_mask = areas < (min_area + tol)
    if not np.any(active_mask):
        active_mask = np.zeros(len(areas), dtype=bool)
        active_mask[np.argmin(areas)] = True
    return -np.mean(areas[active_mask])

def _bary_to_cart(bary):
    """Convert barycentric coords (a,b,c) with a+b+c=1 to cartesian."""
    return bary[:, 0:1] * V0 + bary[:, 1:2] * V1 + bary[:, 2:3] * V2

def _cart_to_bary(pts):
    """Convert cartesian points to barycentric coords."""
    n = len(pts)
    bary = np.zeros((n, 3))
    for i in range(n):
        px, py = pts[i]
        c = 2 * py / np.sqrt(3)
        b = px - 0.5 * c
        a = 1 - b - c
        bary[i] = [a, b, c]
    return bary

def _clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to be non-negative and sum to 1."""
    bary = np.maximum(bary, 0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, eps)
    bary = bary / s
    return bary

def _soft_min_area(pts, beta=50.0):
    """
    Smooth approximation of the minimum triangle area using soft-min.
    Returns (value, gradient) where gradient is w.r.t. the flat point coordinates.
    """
    n = len(pts)
    areas, grad_list = _compute_areas_and_grads(pts)
    
    max_area = np.max(areas)
    exp_vals = np.exp(-beta * (areas - max_area))
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = max_area - log_sum / beta
    
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros((n, 2))
    triples = _all_triples(n)
    for idx in range(len(triples)):
        w = weights[idx]
        if w < 1e-15:
            continue
        i, j, k = triples[idx]
        g_pi, g_pj, g_pk = grad_list[idx]
        grad[i] += w * g_pi
        grad[j] += w * g_pj
        grad[k] += w * g_pk
    
    return soft_min_val, grad

def _objective_bary(bary_flat, beta=50.0):
    """
    Objective in barycentric coordinates using soft-min.
    """
    bary = bary_flat.reshape(-1, 3)
    bary_clamped = _clamp_bary(bary)
    pts = _bary_to_cart(bary_clamped)
    soft_min_val, grad_pts = _soft_min_area(pts, beta)
    
    grad_bary = np.zeros_like(bary_clamped)
    for i in range(len(pts)):
        grad_bary[i, 0] = np.dot(grad_pts[i], V0)
        grad_bary[i, 1] = np.dot(grad_pts[i], V1)
        grad_bary[i, 2] = np.dot(grad_pts[i], V2)
    
    return -soft_min_val, -grad_bary.flatten()

def _objective_bary_value_only(bary_flat, beta=50.0):
    """Value-only objective for Nelder-Mead."""
    bary = bary_flat.reshape(-1, 3)
    bary_clamped = _clamp_bary(bary)
    pts = _bary_to_cart(bary_clamped)
    soft_min_val, _ = _soft_min_area(pts, beta)
    return -soft_min_val

def _make_initial_bary_grid(jitter=0.0, seed=0):
    """Generate structured initial barycentric coordinates."""
    n = N
    rng = np.random.RandomState(seed)
    
    bary = np.zeros((n, 3))
    
    eps_in = 0.08
    bary[0] = [1 - eps_in, eps_in/2, eps_in/2]
    bary[1] = [eps_in/2, 1 - eps_in, eps_in/2]
    bary[2] = [eps_in/2, eps_in/2, 1 - eps_in]
    
    eps_edge = 0.05
    bary[3] = [1 - 2*eps_edge, eps_edge, eps_edge]
    bary[4] = [eps_edge, 1 - 2*eps_edge, eps_edge]
    bary[5] = [eps_edge, eps_edge, 1 - 2*eps_edge]
    
    interior_configs = [
        [1/3, 1/3, 1/3],
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4],
        [0.5, 0.25, 0.25],
    ]
    for i in range(5):
        bary[6 + i] = interior_configs[i]
    
    if jitter > 0:
        noise = rng.uniform(-jitter, jitter, size=bary.shape)
        bary += noise
        bary = _clamp_bary(bary)
    
    return bary

def _make_initial_bary_random(seed=0):
    """Generate random initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
    u = rng.random((N, 1))
    v = rng.random((N, 1))
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    
    bary = np.zeros((N, 3))
    bary[:, 0] = 1 - u[:, 0] - v[:, 0]
    bary[:, 1] = u[:, 0]
    bary[:, 2] = v[:, 0]
    
    bary = _clamp_bary(bary)
    
    return bary

def _make_initial_bary_perturb(base_bary, jitter, seed=0):
    """Perturb a base configuration."""
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-jitter, jitter, size=base_bary.shape)
    bary = base_bary + noise
    bary = _clamp_bary(bary)
    return bary

def _local_optimize_sqp(bary_flat, tol=1e-5, maxiter=500):
    """
    Local optimization using SLSQP with SQP-style active set objective.
    Works in barycentric space with bounds [0,1].
    """
    bounds = [(0, 1)] * (3 * N)
    
    def sqp_obj_bary(bary_flat, tol):
        bary = bary_flat.reshape(-1, 3)
        bary_clamped = _clamp_bary(bary)
        pts = _bary_to_cart(bary_clamped)
        pts_flat = pts.flatten()
        
        n = N
        areas, grad_list = _compute_areas_and_grads(pts)
        
        min_area = np.min(areas)
        active_mask = areas < (min_area + tol)
        if not np.any(active_mask):
            active_mask = np.zeros(len(areas), dtype=bool)
            active_mask[np.argmin(areas)] = True
        
        active_areas = areas[active_mask]
        value = -np.mean(active_areas)
        
        grad_pts = np.zeros((n, 2))
        triples = _all_triples(n)
        active_indices = np.where(active_mask)[0]
        for idx in active_indices:
            i, j, k = triples[idx]
            g_pi, g_pj, g_pk = grad_list[idx]
            grad_pts[i] += g_pi
            grad_pts[j] += g_pj
            grad_pts[k] += g_pk
        
        num_active = len(active_indices)
        if num_active > 0:
            grad_pts /= num_active
        
        # Chain rule to barycentric
        grad_bary = np.zeros_like(bary_clamped)
        for i in range(n):
            grad_bary[i, 0] = np.dot(grad_pts[i], V0)
            grad_bary[i, 1] = np.dot(grad_pts[i], V1)
            grad_bary[i, 2] = np.dot(grad_pts[i], V2)
        
        return value, -grad_bary.flatten()
    
    result = minimize(
        sqp_obj_bary,
        bary_flat,
        method='SLSQP',
        jac=True,
        bounds=bounds,
        args=(tol,),
        options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
    )
    
    return result.x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = -float('inf')
    best_bary = None
    
    # Generate diverse initial configurations
    base_bary = _make_initial_bary_grid(jitter=0.0, seed=0)
    
    initial_configs = []
    initial_configs.append(base_bary.copy())
    
    for j in range(1, 8):
        jitter_amt = 0.05 * j
        config = _make_initial_bary_perturb(base_bary, jitter_amt, seed=j)
        initial_configs.append(config)
    
    for r in range(5):
        config = _make_initial_bary_random(seed=100 + r)
        initial_configs.append(config)
    
    # Phase 1: SLSQP with SQP active-set objective
    for idx, bary_config in enumerate(initial_configs):
        if time.time() > deadline - 40.0:
            break
        
        bary_flat = bary_config.flatten()
        
        try:
            tol_val = 1e-5
            optimized_bary_flat = _local_optimize_sqp(bary_flat, tol=tol_val, maxiter=400)
            
            if time.time() > deadline - 20.0:
                break
            
            bary_opt = _clamp_bary(optimized_bary_flat.reshape(-1, 3))
            pts_opt = _bary_to_cart(bary_opt)
            score = min_triangle_area(pts_opt)
            
            if score > best_score:
                best_score = score
                best_points = pts_opt.copy()
                best_bary = bary_opt.copy()
                
        except Exception:
            pass
        
        # Also try L-BFGS-B with soft-min as a different approach
        if time.time() > deadline - 15.0:
            break
        
        try:
            beta_val = 30.0
            bounds = [(0, 1)] * (3 * N)
            result_l = minimize(
                _objective_bary,
                bary_flat,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                args=(beta_val,),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            
            bary_l = _clamp_bary(result_l.x.reshape(-1, 3))
            pts_l = _bary_to_cart(bary_l)
            score_l = min_triangle_area(pts_l)
            
            if score_l > best_score:
                best_score = score_l
                best_points = pts_l.copy()
                best_bary = bary_l.copy()
                
        except Exception:
            pass
    
    # Phase 2: Polish with higher precision active set
    if best_bary is not None and time.time() < deadline - 15.0:
        try:
            tol_high = 1e-7
            polished_bary_flat = _local_optimize_sqp(best_bary.flatten(), tol=tol_high, maxiter=500)
            bary_polished = _clamp_bary(polished_bary_flat.reshape(-1, 3))
            pts_polished = _bary_to_cart(bary_polished)
            score_polished = min_triangle_area(pts_polished)
            
            if score_polished > best_score:
                best_score = score_polished
                best_points = pts_polished.copy()
                best_bary = bary_polished.copy()
                
        except Exception:
            pass
    
    # Phase 3: Final polish with soft-min high beta
    if best_bary is not None and time.time() < deadline - 8.0:
        try:
            beta_high = 100.0
            bounds = [(0, 1)] * (3 * N)
            result_final = minimize(
                _objective_bary,
                best_bary.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                args=(beta_high,),
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            
            bary_final = _clamp_bary(result_final.x.reshape(-1, 3))
            pts_final = _bary_to_cart(bary_final)
            score_final = min_triangle_area(pts_final)
            
            if score_final > best_score:
                best_score = score_final
                best_points = pts_final.copy()
                best_bary = bary_final.copy()
                
        except Exception:
            pass
    
    # Phase 4: One more SLSQP polish with very tight tolerance
    if best_bary is not None and time.time() < deadline - 5.0:
        try:
            tol_very_tight = 1e-8
            polished_bary_flat = _local_optimize_sqp(best_bary.flatten(), tol=tol_very_tight, maxiter=300)
            bary_polished = _clamp_bary(polished_bary_flat.reshape(-1, 3))
            pts_polished = _bary_to_cart(bary_polished)
            score_polished = min_triangle_area(pts_polished)
            
            if score_polished > best_score:
                best_score = score_polished
                best_points = pts_polished.copy()
                best_bary = bary_polished.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        bary = _make_initial_bary_grid(jitter=0.0, seed=0)
        best_points = _bary_to_cart(bary)
    
    return best_points
# EVOLVE-BLOCK-END
