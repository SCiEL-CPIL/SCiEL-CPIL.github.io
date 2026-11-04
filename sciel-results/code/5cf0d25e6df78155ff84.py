"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute triple indices for efficiency
_triple_indices = np.array(list(combinations(range(N), 3)))
_TRI_I = _triple_indices[:, 0]
_TRI_J = _triple_indices[:, 1]
_TRI_K = _triple_indices[:, 2]

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _triangle_areas(pts):
    """Compute all triangle areas efficiently."""
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_min(ratios, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    shifted = np.clip(shifted, -50, 0)
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return min_r - (1.0 / tau) * log_sum

def _objective(params, tau):
    """
    Objective function: minimize -smooth_min(ratios).
    params: 22 values representing p_3 ... p_13.
    p_1 is fixed at (0,0), p_2 is fixed at (1,0).
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 1e9
    
    areas = _triangle_areas(pts)
    ratios = areas / hull_area
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _transform_to_fixed(pts):
    """Transform points so p[0]=(0,0) and p[1]=(1,0), return params for p[2:]"""
    pts = pts - pts[0]
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        return None
    pts = pts / length
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    return pts[2:].flatten()

def _generate_initial_params(rng, method='polygon'):
    """
    Generate initial parameters for the 22 free coordinates.
    """
    if method == 'polygon':
        # Regular 13-gon with random perturbation
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles += rng.uniform(-0.3, 0.3, N)
        radii = 1.0 + rng.normal(0, 0.15, N)
        
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        params = _transform_to_fixed(pts)
        if params is not None:
            return params
        return rng.uniform(-1, 1, 22)
    
    elif method == 'random':
        return rng.uniform(-2.0, 2.0, 22)
    
    elif method == 'halton':
        # Halton sequence for low-discrepancy
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i, 0] = halton(i + 1, 2)
            pts[i, 1] = halton(i + 1, 3)
        
        # Add small random perturbation
        pts += rng.normal(0, 0.08, (N, 2))
        
        params = _transform_to_fixed(pts)
        if params is not None:
            return params
        return rng.uniform(-1, 1, 22)
    
    elif method == 'concentric':
        # Two concentric rings with varied radii
        n_outer = 7
        n_inner = 6
        outer_angles = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + rng.uniform(-0.15, 0.15, n_outer)
        inner_angles = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner + rng.uniform(-0.15, 0.15, n_inner)
        
        r_outer = 1.0 + rng.normal(0, 0.1, n_outer)
        r_inner = 0.55 + rng.normal(0, 0.08, n_inner)
        
        pts = np.vstack([
            np.column_stack((r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles))),
            np.column_stack((r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)))
        ])
        params = _transform_to_fixed(pts)
        if params is not None:
            return params
        return rng.uniform(-1, 1, 22)
        
    elif method == 'grid':
        # 3x3 grid plus 4 extra points with perturbation
        grid_pts = np.array([
            [0, 0], [0.5, 0], [1, 0],
            [0, 0.5], [0.5, 0.5], [1, 0.5],
            [0, 1], [0.5, 1], [1, 1]
        ])
        extra_pts = np.array([
            [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]
        ])
        pts = np.vstack([grid_pts, extra_pts])
        pts += rng.normal(0, 0.05, (N, 2))
        params = _transform_to_fixed(pts)
        if params is not None:
            return params
        return rng.uniform(-1, 1, 22)

    else:
        return rng.uniform(-2.0, 2.0, 22)

def _local_search(params, tau, max_iter=500, ftol=1e-14):
    """Perform L-BFGS-B local search from given params."""
    try:
        res = minimize(
            _objective,
            params,
            args=(tau,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-10}
        )
        return res.x
    except Exception:
        return params

def _nelder_mead_polish(params, max_iter=2000):
    """Perform Nelder-Mead polish on exact objective."""
    try:
        def _exact_obj(p):
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = p.reshape(N - 2, 2)
            return -_compute_min_ratio(pts)
        
        res = minimize(
            _exact_obj,
            params,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        return res.x
    except Exception:
        return params

def construct():
    start_time = time.time()
    time_limit = 135.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    def _update_best(p, score):
        nonlocal best_params, best_score
        if score > best_score:
            best_score = score
            best_params = p.copy()
    
    # Phase 1: Differential Evolution for global search
    if (time.time() - start_time) < time_limit - 60.0:
        try:
            tau_global = 25.0
            bounds = [(-3.0, 3.0)] * (2 * (N - 2))
            
            de_res = differential_evolution(
                _objective,
                bounds,
                args=(tau_global,),
                maxiter=30,
                popsize=20,
                tol=1e-4,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if de_res.x is not None:
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = de_res.x.reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                _update_best(de_res.x, score)
        except Exception:
            pass
    
    # Phase 2: Generate diverse initial candidates
    candidates = []
    if best_params is not None:
        candidates.append(best_params.copy())
    
    methods = ['polygon', 'halton', 'random', 'concentric', 'grid', 
               'polygon', 'halton', 'concentric', 'random', 'grid']
    for method in methods:
        if (time.time() - start_time) > time_limit - 30.0:
            break
        p = _generate_initial_params(rng, method)
        candidates.append(p)
    
    # Phase 3: Multi-stage annealing with L-BFGS-B
    tau_schedule = [20, 50, 200, 1000, 5000]
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 15.0:
            break
        
        for i in range(len(candidates)):
            if (time.time() - start_time) > time_limit - 5.0:
                break
            
            new_params = _local_search(candidates[i], tau, max_iter=300)
            
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = new_params.reshape(N - 2, 2)
            score = _compute_min_ratio(pts)
            
            _update_best(new_params, score)
            candidates[i] = new_params
    
    # Phase 4: Nelder-Mead polish on best result
    if best_params is not None and (time.time() - start_time) < time_limit - 3.0:
        polished = _nelder_mead_polish(best_params, max_iter=3000)
        pts = np.empty((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = polished.reshape(N - 2, 2)
        score = _compute_min_ratio(pts)
        _update_best(polished, score)
    
    # Phase 5: Final fine polish with very high tau L-BFGS
    if best_params is not None and (time.time() - start_time) < time_limit - 1.0:
        try:
            final_params = _local_search(best_params, 10000, max_iter=500)
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = final_params.reshape(N - 2, 2)
            score = _compute_min_ratio(pts)
            _update_best(final_params, score)
        except Exception:
            pass
    
    # Final result
    if best_params is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = best_params.reshape(N - 2, 2)
    
    return pts

# EVOLVE-BLOCK-END
