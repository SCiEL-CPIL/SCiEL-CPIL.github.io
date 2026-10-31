"""Rewritten program for circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _initial_positions(n, W, H):
    """Generate initial random positions for n circles within [0,W]x[0,H]."""
    pts = np.random.uniform(0.0, 1.0, size=(n, 2))
    return pts * np.array([W, H])

def _compute_radii(pts, W, H):
    """Compute maximum possible radius for each circle given positions.
    r_i = min(distance to each other circle's center minus 0, distance to boundaries)
    Actually r_i is limited by: for each pair (i,j), r_i + r_j <= dist(i,j)
    and r_i <= min(x_i, W - x_i, y_i, H - y_i)
    
    We use a simple iterative approach: start with boundary limits, then reduce based on neighbors.
    """
    n = len(pts)
    # Boundary limits
    r = np.minimum(np.minimum(pts[:, 0], W - pts[:, 0]), np.minimum(pts[:, 1], H - pts[:, 1]))
    r = np.maximum(r, 0.0)
    
    # Iterative relaxation to satisfy pairwise constraints
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                # We need r_i + r_j <= d
                # If violated, reduce the larger one or split
                if r[i] + r[j] > d and d > 1e-10:
                    excess = r[i] + r[j] - d
                    # Reduce proportionally
                    total = r[i] + r[j]
                    if total > 1e-10:
                        r[i] -= excess * (r[i] / total)
                        r[j] -= excess * (r[j] / total)
                    else:
                        r[i] = d / 2
                        r[j] = d / 2
                    changed = True
        if not changed:
            break
    return np.maximum(r, 0.0)

def _objective(params, n, W, H):
    """Objective function to minimize (negative sum of radii)."""
    pts = params.reshape(n, 2)
    r = _compute_radii(pts, W, H)
    return -np.sum(r)

def _penalty_objective(params, n, W, H):
    """Penalty-based objective for optimization."""
    pts = params.reshape(n, 2)
    # Check bounds
    for i in range(n):
        if pts[i, 0] < -0.01 or pts[i, 0] > W + 0.01 or pts[i, 1] < -0.01 or pts[i, 1] > H + 0.01:
            return 1e6
    r = _compute_radii(pts, W, H)
    return -np.sum(r)

def construct():
    """Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    best_sum = 0.0
    best_pts = None
    best_W = 1.0
    best_H = 1.0
    
    # Try different aspect ratios
    # Perimeter = 2(W+H) = 4 => W+H = 2
    # Try W from 0.5 to 1.5 (H = 2 - W)
    aspect_ratios = np.linspace(0.5, 1.5, 21)
    
    for W in aspect_ratios:
        H = 2.0 - W
        if H <= 0.01:
            continue
        
        # Multiple random restarts
        for restart in range(5):
            random.seed(42 + restart * 1000 + int(W * 1000))
            np.random.seed(42 + restart * 1000 + int(W * 1000))
            
            pts = _initial_positions(N, W, H)
            x0 = pts.flatten()
            
            # Use Nelder-Mead for robustness
            try:
                res = minimize(_penalty_objective, x0, args=(N, W, H),
                               method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < 1e5:  # Valid solution
                    pts_opt = res.x.reshape(N, 2)
                    r_opt = _compute_radii(pts_opt, W, H)
                    s = np.sum(r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_pts = pts_opt.copy()
                        best_W = W
                        best_H = H
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    r_best = _compute_radii(best_pts, best_W, best_H)
    # Construct output: [x, y, r]
    result = np.column_stack([best_pts[:, 0], best_pts[:, 1], r_best])
    return result
# EVOLVE-BLOCK-END
