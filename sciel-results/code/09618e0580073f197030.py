"""Heilbronn triangles: 11 points in unit equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import math

N = 11
H = math.sqrt(3) / 2

def _tri_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_area(pts):
    min_a = float('inf')
    for i, j, k in combinations(range(len(pts)), 3):
        a = _tri_area(pts[i], pts[j], pts[k])
        if a < min_a:
            min_a = a
    return min_a

def _project(p, m=1e-9):
    x, y = p
    if y < m:
        y = m
    min_x = y / math.sqrt(3)
    if x < min_x:
        x = min_x
    max_x = 1 - y / math.sqrt(3)
    if x > max_x:
        x = max_x
    if x < m:
        x = m
    if x > 1 - m:
        x = 1 - m
    return [x, y]

def _neg_min_area(flat):
    pts = flat.reshape(N, 2)
    pts = np.array([_project(p) for p in pts])
    return -_min_area(pts)

def _neg_min_area_with_grad(flat):
    """Subgradient-based approach: consider all triangles near the minimum."""
    pts = flat.reshape(N, 2)
    pts = np.array([_project(p) for p in pts])
    
    # Compute all triangle areas
    areas = []
    for i, j, k in combinations(range(N), 3):
        a = _tri_area(pts[i], pts[j], pts[k])
        areas.append((a, i, j, k))
    
    min_a = min(areas[0][0])
    
    # Find all triangles within a small tolerance of the minimum
    tol = max(min_a * 0.01, 1e-10)
    critical = [(a, i, j, k) for a, i, j, k in areas if a < min_a + tol]
    
    if not critical:
        critical = [areas[0]]
    
    grad = np.zeros((N, 2))
    
    for a, i, j, k in critical:
        p1, p2, p3 = pts[i], pts[j], pts[k]
        signed_area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        sign = 1.0 if signed_area > 0 else -1.0
        weight = 1.0 / len(critical)
        
        g_i = np.array([
            -0.5 * sign * (p3[1] - p1[1]) + 0.5 * sign * (p2[1] - p1[1]),
            0.5 * sign * (p2[0] - p1[0]) - 0.5 * sign * (p3[0] - p1[0])
        ])
        g_j = np.array([
            0.5 * sign * (p3[1] - p1[1]),
            -0.5 * sign * (p3[0] - p1[0])
        ])
        g_k = np.array([
            -0.5 * sign * (p2[1] - p1[1]),
            0.5 * sign * (p2[0] - p1[0])
        ])
        
        grad[i] += weight * g_i
        grad[j] += weight * g_j
        grad[k] += weight * g_k
    
    return -min_a, -grad.flatten()

def _neg_min_area_smooth(flat, eps=0.05):
    """Smooth approximation using log-sum-exp for better optimization."""
    pts = flat.reshape(N, 2)
    pts = np.array([_project(p) for p in pts])
    
    areas = np.array([_tri_area(pts[i], pts[j], pts[k]) 
                      for i, j, k in combinations(range(N), 3)])
    
    # Use soft-min approximation
    scaled = -areas / eps
    max_s = np.max(scaled)
    log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    soft_min = -eps * log_sum
    
    return -soft_min

def _neg_min_area_smooth_grad(flat, eps=0.05):
    """Gradient of the smooth approximation."""
    pts = flat.reshape(N, 2)
    pts = np.array([_project(p) for p in pts])
    
    combos = list(combinations(range(N), 3))
    n_combos = len(combos)
    
    areas = np.zeros(n_combos)
    for idx, (i, j, k) in enumerate(combos):
        areas[idx] = _tri_area(pts[i], pts[j], pts[k])
    
    # Soft min: f = -eps * log(sum(exp(-a_i/eps)))
    # df/da_i = (exp(-a_i/eps)) / sum(exp(-a_j/eps))
    scaled = -areas / eps
    max_s = np.max(scaled)
    exp_s = np.exp(scaled - max_s)
    weights = exp_s / np.sum(exp_s)
    
    soft_min_val = -eps * (max_s + np.log(np.sum(exp_s)))
    
    grad = np.zeros((N, 2))
    
    for idx, (i, j, k) in enumerate(combos):
        if weights[idx] < 1e-12:
            continue
        w = weights[idx]
        p1, p2, p3 = pts[i], pts[j], pts[k]
        signed_area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        sign = 1.0 if signed_area > 0 else -1.0
        
        g_i = np.array([
            -0.5 * sign * (p3[1] - p1[1]) + 0.5 * sign * (p2[1] - p1[1]),
            0.5 * sign * (p2[0] - p1[0]) - 0.5 * sign * (p3[0] - p1[0])
        ])
        g_j = np.array([
            0.5 * sign * (p3[1] - p1[1]),
            -0.5 * sign * (p3[0] - p1[0])
        ])
        g_k = np.array([
            -0.5 * sign * (p2[1] - p1[1]),
            0.5 * sign * (p2[0] - p1[0])
        ])
        
        # d(soft_min)/d(point) = -eps * sum_i (weights_i * d(a_i)/d(point))
        # We want to minimize -soft_min, so gradient is soft_min gradient
        grad[i] += w * g_i
        grad[j] += w * g_j
        grad[k] += w * g_k
    
    return -soft_min_val, -grad.flatten()

def construct():
    h = H
    
    # Known good configurations for initialization
    init1 = [
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, 0.0], [0.75, h/2], [0.25, h/2],
        [1/3, h/3], [2/3, h/3], [0.5, 2*h/3],
        [0.5, h/3],
        [0.5, h/6]
    ]
    
    init2 = [
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, 0.0], [0.75, h/2], [0.25, h/2],
        [0.2, 0.2], [0.8, 0.2], [0.5, 0.6],
        [0.33, 0.33],
        [0.67, 0.33]
    ]
    
    init3 = [
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, 0.0], [0.75, h/2], [0.25, h/2],
        [0.15, 0.15], [0.85, 0.15], [0.5, 0.5],
        [0.3, 0.4],
        [0.7, 0.4]
    ]
    
    # Symmetric configuration
    init4 = [
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, 0.0], [0.75, h/2], [0.25, h/2],
        [0.5, h/4], [0.25, h/4], [0.75, h/4],
        [0.5, 3*h/4],
        [0.5, h/2]
    ]
    
    inits = [np.array(init1).flatten(), np.array(init2).flatten(), 
             np.array(init3).flatten(), np.array(init4).flatten()]
    
    bounds = [(0, 1), (0, h)] * N
    
    best_flat = None
    best_val = float('inf')
    
    # Phase 1: Differential evolution with multiple seeds
    for seed in [42, 123, 777]:
        for init_flat in inits:
            try:
                res_de = differential_evolution(
                    _neg_min_area, bounds, seed=seed, maxiter=80, popsize=15,
                    tol=1e-7, init='sobol', polish=True, updating='deferred',
                    x0=init_flat
                )
                if res_de.fun < best_val:
                    best_val = res_de.fun
                    best_flat = res_de.x.copy()
            except Exception:
                pass
    
    # Phase 2: Polish with smooth gradient-based method
    if best_flat is not None:
        for eps in [0.02, 0.01, 0.005, 0.001]:
            try:
                res_smooth = minimize(_neg_min_area_smooth_grad, best_flat, 
                                       args=(eps,), method='L-BFGS-B',
                                       options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-11})
                if res_smooth.fun < best_val:
                    best_val = res_smooth.fun
                    best_flat = res_smooth.x.copy()
            except Exception:
                pass
        
        # Nelder-Mead polish
        try:
            res_nm = minimize(_neg_min_area, best_flat, method='Nelder-Mead',
                              options={'maxiter': 15000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_flat = res_nm.x.copy()
        except Exception:
            pass
        
        # Subgradient L-BFGS-B
        try:
            res_lbfgs = minimize(_neg_min_area_with_grad, best_flat, method='L-BFGS-B',
                                 options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_flat = res_lbfgs.x.copy()
        except Exception:
            pass
    
    # Phase 3: Random restarts with better sampling
    rng = np.random.RandomState(123)
    for trial in range(15):
        pts = []
        for _ in range(N):
            # Stratified sampling with jitter
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            p = [u + 0.5 * v, h * v]
            pts.append(_project(p))
        
        flat0 = np.array(pts).flatten()
        try:
            # Quick smooth optimization
            res_s = minimize(_neg_min_area_smooth_grad, flat0, args=(0.01,), 
                             method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res_s.fun < best_val:
                best_val = res_s.fun
                best_flat = res_s.x.copy()
            else:
                # Try Nelder-Mead from this start
                res_nm = minimize(_neg_min_area, flat0, method='Nelder-Mead',
                                  options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
                if res_nm.fun < best_val:
                    best_val = res_nm.fun
                    best_flat = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final intensive polish
    if best_flat is not None:
        # Multiple smooth epsilons
        for eps in [0.005, 0.002, 0.001, 0.0005]:
            try:
                res_smooth = minimize(_neg_min_area_smooth_grad, best_flat, 
                                       args=(eps,), method='L-BFGS-B',
                                       options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13})
                if res_smooth.fun < best_val:
                    best_val = res_smooth.fun
                    best_flat = res_smooth.x.copy()
            except Exception:
                pass
        
        # Final Nelder-Mead
        try:
            res_nm = minimize(_neg_min_area, best_flat, method='Nelder-Mead',
                              options={'maxiter': 30000, 'xatol': 1e-13, 'fatol': 1e-15})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_flat = res_nm.x.copy()
        except Exception:
            pass
        
        # Final subgradient
        try:
            res_lbfgs = minimize(_neg_min_area_with_grad, best_flat, method='L-BFGS-B',
                                 options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_flat = res_lbfgs.x.copy()
        except Exception:
            pass
    
    points = best_flat.reshape(N, 2)
    points = np.array([_project(p) for p in points])
    return points
