import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 14

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized min triangle area
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def fast_score(points):
    """
    Fast score calculation for inner optimization loops.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def local_refine_powell(points, deadline, max_iter=500):
    """
    Local refinement using Powell's method, which handles non-smooth objectives better.
    """
    current_points = points.copy()
    current_score = get_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -fast_score(pts)
    
    try:
        res = minimize(objective, current_points.flatten(), method='Powell', 
                      options={'maxiter': max_iter, 'xtol': 1e-8, 'ftol': 1e-10})
        
        if time.time() < deadline - 2.0:
            if -res.fun > best_score:
                best_score = -res.fun
                best_points = res.x.reshape(N, 2)
    except Exception:
        pass
    
    return best_points, best_score

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We keep a safety margin.
    deadline = start_time + 145.0
    
    # Initial guess: Regular N-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    initial_points = np.column_stack((x, y))
    
    best_points = initial_points.copy()
    best_score = get_score(best_points)
    
    # Objective function for minimizers (maximize score -> minimize -score)
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -fast_score(pts)
        
    # 1. Global Search using Differential Evolution
    bounds = [(-2, 2)] * (2 * N)
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=30,
            popsize=12,
            tol=0.01,
            polish=False
        )
        
        if time.time() < deadline - 30.0:
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(N, 2)
                
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 30.0:
        return best_points

    # 2. Local Refinement using Powell's method
    best_points, best_score = local_refine_powell(best_points, deadline)
    
    # Check time
    if time.time() > deadline - 10.0:
        return best_points

    # 3. Second round of DE from the refined point (if time allows)
    if time.time() < deadline - 20.0:
        try:
            center = best_points.mean(axis=0)
            spread = np.std(best_points, axis=0)
            tight_bounds = []
            for i in range(2 * N):
                idx = i // 2
                coord = i % 2
                lower = best_points[idx, coord] - 2 * spread[coord] - 0.5
                upper = best_points[idx, coord] + 2 * spread[coord] + 0.5
                tight_bounds.append((lower, upper))
            
            res2 = differential_evolution(
                objective, 
                tight_bounds, 
                seed=123,
                maxiter=20,
                popsize=10,
                tol=0.01,
                polish=False
            )
            
            if time.time() < deadline - 10.0:
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_points = res2.x.reshape(N, 2)
                    
        except Exception:
            pass
    
    # Final local refinement
    if time.time() < deadline - 5.0:
        best_points, best_score = local_refine_powell(best_points, deadline)

    return best_points
