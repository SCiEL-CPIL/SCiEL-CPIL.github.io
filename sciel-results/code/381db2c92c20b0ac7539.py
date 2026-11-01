import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import sys

N = 14

# Precompute triangle indices
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        # Check for NaN/Inf
        if not np.all(np.isfinite(points)):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def soft_min_score(params, tau):
    """
    Objective for L-BFGS-B.
    params: [r1, theta1, r2, theta2] for two rings of 7 points each.
    Returns -soft_min(areas) / hull_area.
    """
    r1, th1, r2, th2 = params
    
    # Generate points for Ring 1 (7 points)
    angles1 = th1 + np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts1 = np.column_stack((r1*np.cos(angles1), r1*np.sin(angles1)))
    
    # Generate points for Ring 2 (7 points)
    angles2 = th2 + np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts2 = np.column_stack((r2*np.cos(angles2), r2*np.sin(angles2)))
    
    points = np.vstack((pts1, pts2))
    
    # Check validity
    if not np.all(np.isfinite(points)):
        return 1e6
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 1e6
    except Exception:
        return 1e6
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min approximation
    areas = np.maximum(areas, 1e-12)
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    score = soft_min_val / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Initial guess based on knowledge: Two nested heptagons
    # Outer ring radius ~1.0, Inner ring radius ~0.5-0.7
    # Offset inner ring by half step (pi/7) to maximize separation
    best_params = np.array([1.0, 0.0, 0.6, np.pi/7])
    best_score = -1e9
    
    def evaluate_params(params):
        nonlocal best_score
        r1, th1, r2, th2 = params
        angles1 = th1 + np.linspace(0, 2*np.pi, 7, endpoint=False)
        pts1 = np.column_stack((r1*np.cos(angles1), r1*np.sin(angles1)))
        angles2 = th2 + np.linspace(0, 2*np.pi, 7, endpoint=False)
        pts2 = np.column_stack((r2*np.cos(angles2), r2*np.sin(angles2)))
        points = np.vstack((pts1, pts2))
        
        if not np.all(np.isfinite(points)):
            return -1e9, points
            
        s = compute_score(points)
        if s > best_score:
            best_score = s
        return s, points
    
    # Initial evaluation
    s0, p0 = evaluate_params(best_params)
    if s0 > 0:
        best_points = p0.copy()
    else:
        # Fallback to regular 14-gon if symmetric init fails
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = compute_score(best_points)

    # Optimization Phase 1: Refine symmetric structure with L-BFGS-B
    # Use a schedule of decreasing tau for soft-min
    taus = [0.05, 0.01, 0.001, 0.0001]
    current_params = best_params.copy()
    
    for tau in taus:
        if time.time() - start_time > deadline - 10:
            break
            
        try:
            res = minimize(
                soft_min_score,
                current_params,
                args=(tau,),
                method='L-BFGS-B',
                bounds=[(0.1, 2.0), (0, 2*np.pi), (0.0, 2.0), (0, 2*np.pi)],
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < soft_min_score(current_params, tau):
                new_s, new_p = evaluate_params(res.x)
                if new_s > best_score:
                    best_score = new_s
                    best_points = new_p.copy()
                    current_params = res.x.copy()
        except Exception:
            pass

    # Optimization Phase 2: Try a few different initial offsets for the inner ring
    # The optimal offset might not be exactly pi/7
    offsets_to_try = [0.0, np.pi/14, np.pi/7, 3*np.pi/14, np.pi/4]
    radii_to_try = [0.5, 0.6, 0.7, 0.8]
    
    for r2 in radii_to_try:
        for th2 in offsets_to_try:
            if time.time() - start_time > deadline - 5:
                break
                
            init_params = np.array([1.0, 0.0, r2, th2])
            # Quick refine with small tau
            try:
                res = minimize(
                    soft_min_score,
                    init_params,
                    args=(0.001,),
                    method='L-BFGS-B',
                    bounds=[(0.5, 1.5), (0, 2*np.pi), (0.1, 1.5), (0, 2*np.pi)],
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                new_s, new_p = evaluate_params(res.x)
                if new_s > best_score:
                    best_score = new_s
                    best_points = new_p.copy()
            except Exception:
                pass

    # Final validation
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Absolute fallback: Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
