import numpy as np
import time
from scipy.optimize import minimize

N = 13

def _triangle_areas(points):
    """Compute all triangle areas for a set of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = points[i], points[j], points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas.append(area)
    return np.array(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _min_triangle_area_ratio(points):
    """Compute the ratio of minimum triangle area to convex hull area."""
    areas = _triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_min_triangle_area_ratio(points, temperature=0.01):
    """
    Smooth surrogate for the minimum triangle area ratio using log-sum-exp.
    This makes the objective differentiable for gradient-based optimization.
    """
    areas = _triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    # Use soft-min: -log-sum-exp(-areas/temperature) * temperature
    neg_areas = -areas / temperature
    max_neg = np.max(neg_areas)
    soft_min = -temperature * (max_neg + np.log(np.sum(np.exp(neg_areas - max_neg))))
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    
    return soft_min / hull_area

def _negative_soft_min_ratio(points_flat, temperature=0.01):
    """Negative of soft-min ratio for minimization."""
    points = points_flat.reshape(N, 2)
    return -_soft_min_triangle_area_ratio(points, temperature)

def _negative_exact_ratio(points_flat):
    """Negative of exact min ratio for minimization."""
    points = points_flat.reshape(N, 2)
    return -_min_triangle_area_ratio(points)

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs 13 points to maximize the minimum triangle area relative to convex hull area.
    Starts from regular 13-gon and uses gradient-based optimization with smooth surrogate,
    then polishes with exact local optimization.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initialize with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = points.copy()
    best_score = _min_triangle_area_ratio(points)
    
    # Phase 1: Coarse optimization with larger temperature (smoother landscape)
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for temp in temperatures:
        if time.time() > deadline:
            break
        
        x0 = points.flatten()
        
        try:
            result = minimize(
                _negative_soft_min_ratio,
                x0,
                args=(temp,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -best_score:
                new_points = result.x.reshape(N, 2)
                new_score = _min_triangle_area_ratio(new_points)
                
                if new_score > best_score:
                    best_points = new_points.copy()
                    best_score = new_score
                    points = new_points.copy()
        except Exception:
            pass
    
    # Phase 2: Fine-tune with exact objective using local search
    if time.time() < deadline:
        x0 = best_points.flatten()
        
        try:
            result = minimize(
                _negative_exact_ratio,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < -best_score:
                new_points = result.x.reshape(N, 2)
                new_score = _min_triangle_area_ratio(new_points)
                
                if new_score > best_score:
                    best_points = new_points.copy()
                    best_score = new_score
        except Exception:
            pass
    
    # Phase 3: Final polish with very tight tolerance
    if time.time() < deadline - 5:
        x0 = best_points.flatten()
        
        try:
            result = minimize(
                _negative_exact_ratio,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if result.fun < -best_score:
                new_points = result.x.reshape(N, 2)
                new_score = _min_triangle_area_ratio(new_points)
                
                if new_score > best_score:
                    best_points = new_points.copy()
                    best_score = new_score
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
