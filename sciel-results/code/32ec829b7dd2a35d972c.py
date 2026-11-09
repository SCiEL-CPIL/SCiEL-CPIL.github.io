# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def _precompute_indices(n):
    """Precompute triangle indices for efficiency."""
    return np.array(list(itertools.combinations(range(n), 3)))

def _compute_dets(points, idx):
    """Compute 2 * signed area (determinant) for all triangles."""
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return det

def _compute_hull_area(points):
    """Compute convex hull area safely."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
        
    A_hull = _compute_hull_area(points)
    if A_hull <= 1e-12:
        return 0.0
        
    idx = _precompute_indices(n)
    dets = _compute_dets(points, idx)
    A_min = 0.5 * np.min(np.abs(dets))
    
    return A_min / A_hull

def _soft_min_penalty(dets, A_hull, t, beta):
    """
    Compute a smooth penalty that approximates the constraint:
    t * A_hull >= |det_k| for all k.
    
    We want to minimize t subject to max_k (|det_k| / (t * A_hull) - 1) <= 0.
    Let g_k = |det_k| / (t * A_hull). We want max(g_k) <= 1.
    Penalty = log(sum(exp(beta * (g_k - 1)))) / beta.
    This is 0 if all g_k <= 1, and positive otherwise, growing smoothly.
    """
    if t <= 1e-12 or A_hull <= 1e-12:
        return 1e6
    g = np.abs(dets) / (t * A_hull)
    # To avoid overflow, shift by max
    g_shifted = g - 1.0
    max_g = np.max(g_shifted)
    if max_g > 50: # Overflow protection
        return max_g / beta + 1.0
    log_sum = np.log(np.sum(np.exp(beta * g_shifted)))
    return (log_sum + max_g * beta) / beta  # Wait, standard log-sum-exp is log(sum(exp(x))) = max(x) + log(sum(exp(x-max(x))))
    # Let's use the standard stable log-sum-exp formula
    # LSE(x) = max(x) + log(sum(exp(x - max(x))))
    # Penalty = LSE(beta * (g - 1)) / beta
    # If all g <= 1, g-1 <= 0, max is <= 0.
    # Actually, simpler: we want to push t up if any constraint is violated.
    # Let's use a different formulation:
    # Objective: t + lambda * max(0, max_k (|det_k|/(t*A_hull) - 1))^2
    # Or simply minimize: t + sum( max(0, |det_k| - t*A_hull)^2 )
    # This is a smooth approximation if we use a soft-max or just a quadratic penalty.
    # Let's stick to a robust quadratic penalty for the constraints.
    # Constraint: |det_k| - t * A_hull <= 0
    # Violation: max(0, |det_k| - t * A_hull)
    # Penalty: sum(violation^2)
    
    viol = np.maximum(0, np.abs(dets) - t * A_hull)
    return np.sum(viol ** 2)

def _epigraph_objective(params, idx, beta=100.0):
    """
    Objective for the epigraph formulation.
    params: [x_0, y_0, ..., x_12, y_12, t]
    We want to minimize t such that t >= max_k (|det_k| / A_hull).
    This is equivalent to minimizing t subject to |det_k| - t * A_hull <= 0 for all k.
    We use a penalty method: f = t + beta * sum( max(0, |det_k| - t * A_hull)^2 )
    """
    n = 13
    pts = params[:2*n].reshape((n, 2))
    t = params[2*n]
    
    if t <= 1e-9:
        t = 1e-9
    
    A_hull = _compute_hull_area(pts)
    if A_hull <= 1e-12:
        return 1e6
    
    dets = _compute_dets(pts, idx)
    
    # Penalty for constraint violation
    # We want t * A_hull >= |det_k|
    # Violation is max(0, |det_k| - t * A_hull)
    viol = np.maximum(0, np.abs(dets) - t * A_hull)
    penalty = np.sum(viol ** 2)
    
    # The objective is t + penalty
    # Note: t is the upper bound on the normalized min area.
    # Minimizing t drives the solution to the tightest possible bound.
    return t + beta * penalty

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    idx = _precompute_indices(n)
    
    # Initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric polygons (1 center, 5 inner, 7 outer)
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    inits.append(np.vstack([center, inner_pts, outer_pts]))
    
    # 3. Random
    rng = np.random.default_rng(seed=42)
    inits.append(rng.random((n, 2)) * 2 - 1)
    
    # 4. Grid
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:13]
    pts += rng.random((13, 2)) * 0.05
    inits.append(pts)
    
    # 5. Hexagonal lattice snippet
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / (np.abs(hex_pts).max() + 1e-9)
    hex_pts = hex_pts[:13]
    inits.append(hex_pts)
    
    # 6. Fibonacci spiral
    angles_fib = np.arange(n) * 2 * np.pi * 0.6180339887
    radii_fib = np.sqrt(np.arange(1, n + 1) / n)
    inits.append(np.column_stack([radii_fib * np.cos(angles_fib), radii_fib * np.sin(angles_fib)]))

    # We will run the epigraph optimization for each init
    # And also keep a Nelder-Mead fallback as it was robust
    
    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Phase 1: Epigraph Optimization
        # Variables: 26 coords + 1 t
        # Initial t: estimate from current score
        current_score = compute_score(init_pts)
        if current_score <= 0:
            current_score = 1e-3
        # t is the normalized min area (A_min / A_hull)
        # So t_init = current_score
        t_init = current_score
        
        x0 = np.concatenate([init_pts.flatten(), [t_init]])
        
        # Use L-BFGS-B for the epigraph formulation
        # It handles the smooth penalty well
        try:
            res = minimize(
                lambda x: _epigraph_objective(x, idx, beta=50.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            final_params = res.x
            final_pts = final_params[:2*n].reshape((n, 2))
            final_score = compute_score(final_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
                
        except Exception:
            pass
        
        # Phase 2: Nelder-Mead Polish on the best result so far (or this init)
        # NM is good for the final non-smooth steps
        base_pts = final_pts if 'final_pts' in locals() else init_pts
        # If epigraph failed or didn't improve, use the init or previous best
        if best_points is not None and compute_score(best_points) > compute_score(base_pts):
            base_pts = best_points
        else:
            base_pts = base_pts.copy()
            
        def nm_objective(x):
            pts = x.reshape((n, 2))
            s = compute_score(pts)
            if s == 0.0:
                return 1.0
            return -s
        
        try:
            res_nm = minimize(nm_objective, base_pts.flatten(), method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            final_pts_nm = res_nm.x.reshape((n, 2))
            final_score_nm = compute_score(final_pts_nm)
            
            if final_score_nm > best_score:
                best_score = final_score_nm
                best_points = final_pts_nm.copy()
                
        except Exception:
            pass

    # Fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
