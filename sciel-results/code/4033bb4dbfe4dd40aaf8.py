"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
import random

def _halton_sequence(n, base):
    """Generate n Halton sequence values in [0, 1) for a given base."""
    seq = np.zeros(n)
    for i in range(n):
        f = 1.0
        i_val = i + 1
        while i_val > 0:
            f /= base
            i_val, rem = divmod(i_val, base)
            seq[i] += rem * f
    return seq

def _generate_halton_points(n_points, seed=1):
    """Generate n_points in 2D using Halton sequences for better spread."""
    # Use two different bases for x and y
    x = _halton_sequence(n_points, 2)
    y = _halton_sequence(n_points, 3)
    
    # Shift to center around origin
    x = x - 0.5
    y = y - 0.5
    
    # Scale to unit radius roughly
    pts = np.column_stack((x, y))
    # Normalize to have max norm 1
    norms = np.linalg.norm(pts, axis=1)
    if np.max(norms) > 0:
        pts = pts / np.max(norms)
    
    return pts

def _compute_min_tri_area_norm(pts):
    """
    Compute the minimum area of any triangle formed by pts,
    normalized by the area of the convex hull of pts.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-9:
            return -1e9
    except Exception:
        return -1e9

    # Compute all triangle areas
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                  p2[0] * (p3[1] - p1[1]) + 
                                  p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area

    return min_area / hull_area

def _soft_min_triangle_ratio(pts, beta=100.0):
    """
    Compute a smooth approximation of the minimum triangle area ratio.
    Uses log-sum-exp trick for soft-min.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                  p2[0] * (p3[1] - p1[1]) + 
                                  p3[0] * (p1[1] - p2[1]))
                areas.append(area)
    
    areas = np.array(areas)
    
    # Soft-min using log-sum-exp
    # min(areas) ≈ (1/beta) * log(sum(exp(-beta * areas))) ... wait, that's not right.
    # Soft-min(x) = -1/beta * log(sum(exp(-beta * x)))
    # So we want to maximize soft-min, which means minimizing -soft-min.
    
    # To avoid overflow, subtract max
    max_neg = np.max(-beta * areas)
    log_sum = max_neg + np.log(np.sum(np.exp(-beta * areas - max_neg)))
    soft_min_val = -log_sum / beta
    
    return soft_min_val / hull_area

def _objective_flat(params, beta=100.0):
    """
    Objective function for minimizer.
    Minimizes -soft_min_triangle_ratio.
    """
    pts = params.reshape(N, 2)
    val = _soft_min_triangle_ratio(pts, beta=beta)
    return -val

def _construct_single_start(p0, start_time, time_limit, beta_schedule=None):
    """
    Run optimization from a single starting point with a schedule of beta values.
    Returns best points found.
    """
    best_score = _compute_min_tri_area_norm(p0.reshape(N, 2))
    best_pts = p0.copy()
    
    if beta_schedule is None:
        beta_schedule = [10.0, 50.0, 100.0, 200.0]
    
    current_pts = p0.copy()
    
    for beta in beta_schedule:
        if (time.time() - start_time) > time_limit - 5.0:
            break
            
        def objective(params):
            return _objective_flat(params, beta=beta)
        
        try:
            res = minimize(
                objective, 
                current_pts, 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'maxfun': 50000, 'disp': False}
            )
            
            if res.success:
                score = _compute_min_tri_area_norm(res.x.reshape(N, 2))
                if score > best_score:
                    best_score = score
                    best_pts = res.x.copy()
                current_pts = res.x.copy()
                
        except Exception:
            pass
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s kill limit
    
    # Generate diverse starting points
    starts = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.cos(angles)
    y0 = np.sin(angles)
    p0 = np.column_stack((x0, y0)).flatten()
    starts.append(p0)
    
    # 2. Halton sequence points
    halton_pts = _generate_halton_points(N)
    starts.append(halton_pts.flatten())
    
    # 3. Perturbed regular polygons
    rng = np.random.default_rng(42)
    for _ in range(10):
        noise = rng.normal(0, 0.2, size=N*2)
        p_pert = p0 + noise
        p_pert = p_pert - np.mean(p_pert, axis=0)
        starts.append(p_pert)
    
    best_pts = None
    best_score = -1e9
    
    # Try each start point
    for i, start in enumerate(starts):
        if (time.time() - start_time) > time_limit - 10.0:
            break
            
        pts, score = _construct_single_start(start, start_time, time_limit)
        
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Final high-precision polish pass on the best solution
    if best_pts is not None and (time.time() - start_time) < time_limit - 10.0:
        # Use a very high beta for the final polish
        def objective_polish(params):
            return _objective_flat(params, beta=500.0)
        
        try:
            res = minimize(
                objective_polish, 
                best_pts, 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'maxfun': 100000, 'disp': False}
            )
            
            if res.success:
                score = _compute_min_tri_area_norm(res.x.reshape(N, 2))
                if score > best_score:
                    best_score = score
                    best_pts = res.x.copy()
                    
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback to regular polygon
        return p0.reshape(N, 2)
        
    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
