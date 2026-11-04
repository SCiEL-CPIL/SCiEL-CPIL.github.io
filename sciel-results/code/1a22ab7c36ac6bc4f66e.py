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

def _inside(p, m=1e-9):
    x, y = p
    if y < -m or x < -m or x > 1 + m:
        return False
    if y > math.sqrt(3) * x + m:
        return False
    if y > math.sqrt(3) * (1 - x) + m:
        return False
    return True

def _project(p, m=1e-8):
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
    pts = flat.reshape(N, 2)
    pts = np.array([_project(p) for p in pts])
    
    # Compute min area and which triangle achieves it
    min_a = float('inf')
    min_idx = (0, 0, 0)
    for i, j, k in combinations(range(N), 3):
        a = _tri_area(pts[i], pts[j], pts[k])
        if a < min_a:
            min_a = a
            min_idx = (i, j, k)
    
    # Compute gradient w.r.t. the minimum triangle
    i, j, k = min_idx
    p1, p2, p3 = pts[i], pts[j], pts[k]
    
    signed_area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    sign = 1.0 if signed_area > 0 else -1.0
    
    grad = np.zeros((N, 2))
    
    grad[i, 0] = -0.5 * sign * (p3[1] - p1[1]) + 0.5 * sign * (p2[1] - p1[1])
    grad[i, 1] = 0.5 * sign * (p2[0] - p1[0]) - 0.5 * sign * (p3[0] - p1[0])
    
    grad[j, 0] = 0.5 * sign * (p3[1] - p1[1])
    grad[j, 1] = -0.5 * sign * (p3[0] - p1[0])
    
    grad[k, 0] = -0.5 * sign * (p2[1] - p1[1])
    grad[k, 1] = 0.5 * sign * (p2[0] - p1[0])
    
    return -min_a, -grad.flatten()

def construct():
    h = H
    
    # New Strategy: Symmetric Parameterization
    # We assume the optimal configuration has 3-fold symmetry or is close to it.
    # We will generate initial guesses based on symmetric patterns.
    
    # Pattern 1: 3 Vertices, 3 Edge Midpoints, 5 Interior (Center + 4 Ring)
    # Vertices: (0,0), (1,0), (0.5, h)
    # Midpoints: (0.5, 0), (0.75, h/2), (0.25, h/2)
    # Interior: Center (0.5, h/3) and 4 points on a circle around it
    
    inits = []
    
    # Init 1: Standard Symmetric
    p1 = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, 0.0], [0.75, h/2], [0.25, h/2],
        [0.5, h/3],
        [0.5 + 0.1, h/3 + 0.1],
        [0.5 - 0.1, h/3 + 0.1],
        [0.5 + 0.1, h/3 - 0.1],
        [0.5 - 0.1, h/3 - 0.1]
    ])
    inits.append(p1.flatten())
    
    # Init 2: Shifted Edge Points
    p2 = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.4, 0.0], [0.8, h/2], [0.2, h/2], # Asymmetric edge points
        [0.5, h/3],
        [0.5 + 0.15, h/3 + 0.15],
        [0.5 - 0.15, h/3 + 0.15],
        [0.5 + 0.15, h/3 - 0.15],
        [0.5 - 0.15, h/3 - 0.15]
    ])
    inits.append(p2.flatten())

    # Init 3: Dense Interior
    p3 = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [0.5, 0.0], [0.75, h/2], [0.25, h/2],
        [0.5, h/3],
        [0.5 + 0.05, h/3 + 0.05],
        [0.5 - 0.05, h/3 + 0.05],
        [0.5 + 0.05, h/3 - 0.05],
        [0.5 - 0.05, h/3 - 0.05]
    ])
    inits.append(p3.flatten())

    bounds = [(0, 1), (0, h)] * N
    
    best_flat = None
    best_val = float('inf')
    
    # Optimize each initial guess
    for init_flat in inits:
        try:
            # Use L-BFGS-B directly for speed, then DE for global search
            res_lbfgs = minimize(_neg_min_area_with_grad, init_flat, method='L-BFGS-B',
                                 options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_flat = res_lbfgs.x
                
            # Then polish with DE from this point? No, DE is global.
            # Let's do a local DE around the best point found so far?
            # Actually, let's just run a standard DE with this init.
            res_de = differential_evolution(
                _neg_min_area, bounds, seed=42, maxiter=100, popsize=15,
                tol=1e-8, init='sobol', polish=True, updating='deferred',
                x0=init_flat
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_flat = res_de.x
                
        except Exception:
            pass

    # Final Polish with Nelder-Mead and L-BFGS-B
    if best_flat is not None:
        try:
            res_nm = minimize(_neg_min_area, best_flat, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_flat = res_nm.x
        except Exception:
            pass
        
        try:
            res_lbfgs = minimize(_neg_min_area_with_grad, best_flat, method='L-BFGS-B',
                                 options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-13})
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_flat = res_lbfgs.x
        except Exception:
            pass
            
    points = best_flat.reshape(N, 2)
    points = np.array([_project(p) for p in points])
    return points
