import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _triangle_areas(points):
    """Compute areas of all triangles formed by points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                areas.append(area)
    return np.array(areas)

def _convex_hull_area_smooth(points):
    """
    Approximate convex hull area assuming points are in convex position 
    and sorted by angle around centroid. 
    This is differentiable if the order doesn't change.
    """
    centroid = np.mean(points, axis=0)
    # Angles relative to centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    # Shoelace formula
    x = sorted_points[:, 0]
    y = sorted_points[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return area

def _soft_min(areas, t):
    """Soft minimum of areas."""
    # Use log-sum-exp trick for stability
    min_val = np.min(areas)
    exp_terms = np.exp(-t * (areas - min_val))
    log_sum = np.log(np.sum(exp_terms))
    return min_val - np.log(len(areas)) / t + np.log(exp_terms).sum() / t # Wait, standard soft min is -1/t log(sum(exp(-t*x)))
    # Correct soft min:
    # S = -1/t * log( sum( exp(-t * x_i) ) )
    # For stability:
    # S = min_x - 1/t * log( sum( exp(-t * (x_i - min_x)) ) )
    return min_val - (1.0 / t) * np.log(np.sum(np.exp(-t * (areas - min_val))))

def _objective(params, t):
    """
    Objective to minimize: - (SoftMin(TriangleAreas) / HullArea)
    """
    points = params.reshape((-1, 2))
    areas = _triangle_areas(points)
    hull_area = _convex_hull_area_smooth(points)
    
    if hull_area < 1e-10:
        return 1e10
        
    soft_min_area = _soft_min(areas, t)
    
    # We want to maximize soft_min_area / hull_area
    # So minimize - (soft_min_area / hull_area)
    return -soft_min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_score = -1e10
    best_points = initial_points
    
    # Annealing schedule for soft-min temperature t
    # Start with low t (smoother), increase t (sharper)
    t_values = [5.0, 20.0, 100.0, 500.0]
    
    current_points = initial_points.copy()
    
    for t in t_values:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _objective,
                current_points.flatten(),
                args=(t,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            
            if res.success or res.fun < best_score:
                candidate_points = res.x.reshape((-1, 2))
                # Calculate actual score for this candidate
                areas = _triangle_areas(candidate_points)
                # Use real convex hull area for final score check
                try:
                    hull = ConvexHull(candidate_points)
                    real_hull_area = hull.volume # In 2D, volume is area
                except Exception:
                    real_hull_area = _convex_hull_area_smooth(candidate_points)
                
                real_min_area = np.min(areas)
                real_score = real_min_area / real_hull_area
                
                if real_score > best_score:
                    best_score = real_score
                    best_points = candidate_points
                current_points = candidate_points
        except Exception as e:
            pass
            
    # Final polish with Nelder-Mead if time permits? 
    # Nelder-Mead is robust but slow. Let's stick to L-BFGS-B results.
    # Maybe one more L-BFGS-B with high t if time remains?
    
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _objective,
                current_points.flatten(),
                args=(1000.0,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            candidate_points = res.x.reshape((-1, 2))
            areas = _triangle_areas(candidate_points)
            try:
                hull = ConvexHull(candidate_points)
                real_hull_area = hull.volume
            except Exception:
                real_hull_area = _convex_hull_area_smooth(candidate_points)
            
            real_min_area = np.min(areas)
            real_score = real_min_area / real_hull_area
            
            if real_score > best_score:
                best_score = real_score
                best_points = candidate_points
        except Exception:
            pass

    return best_points
