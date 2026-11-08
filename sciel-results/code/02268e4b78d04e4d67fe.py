import numpy as np

N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _all_triangle_areas(points):
    """Compute all triangle areas vectorized."""
    n = len(points)
    # Precompute for speed
    areas = np.empty((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas[i, j, k] = area
                areas[j, i, k] = area
                areas[j, k, i] = area
                areas[i, k, j] = area
                areas[k, i, j] = area
                areas[k, j, i] = area
    return areas

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 1.0

def _score(points):
    """Compute score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(x, temperature=0.01):
    """Soft minimum objective for smoother optimization."""
    points = x.reshape(N, 2)
    n = N
    # Compute all triangle areas
    log_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                # Soft min: -log(sum(exp(-area/T))) * T
                log_sum += -area / temperature
                count += 1
    # This is an approximation; use proper soft min
    # Better: use the actual min but with smoothing
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    score = min_area / hull_area
    return -score

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def construct():
    """
    Constructs 13 points to maximize the minimum triangle area relative to hull area.
    Uses multiple optimization strategies starting from regular polygon.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s deadline for safety (150s limit, 35s margin)
    
    # Start with regular 13-gon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Strategy 1: Nelder-Mead from regular polygon
    x0 = best_points.flatten()
    try:
        result = minimize(_soft_min_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 
                                  'adaptive': True, 'maxfev': 5000})
        
        if time.time() < deadline:
            candidate_points = result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
    except Exception:
        pass
    
    # Strategy 2: COBYLA from regular polygon
    if time.time() < deadline - 20:
        x0 = best_points.flatten()
        try:
            result = minimize(_soft_min_objective, x0, method='COBYLA',
                             options={'maxiter': 1000, 'rhobeg': 0.1})
            
            if time.time() < deadline:
                candidate_points = result.x.reshape(N, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
        except Exception:
            pass
    
    # Strategy 3: Multiple Nelder-Mead restarts with perturbation
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for restart in range(10):
            if time.time() >= deadline - 15:
                break
            
            # Different perturbation scales
            scale = 0.05 * (restart + 1)
            perturbation = rng.normal(0, scale, (N, 2))
            x0 = (best_points + perturbation).flatten()
            
            try:
                result = minimize(_soft_min_objective, x0, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 
                                          'adaptive': True, 'maxfev': 3000})
                
                candidate_points = result.x.reshape(N, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
            except Exception:
                pass
    
    # Strategy 4: Try slightly different initial configurations
    if time.time() < deadline - 20:
        # Try regular polygon with slight radial perturbation
        rng = np.random.default_rng(123)
        for _ in range(3):
            if time.time() >= deadline - 10:
                break
            
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radii = 1.0 + rng.normal(0, 0.05, N)
            points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
            x0 = points.flatten()
            
            try:
                result = minimize(_soft_min_objective, x0, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 
                                          'adaptive': True, 'maxfev': 3000})
                
                candidate_points = result.x.reshape(N, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
            except Exception:
                pass
    
    # Final polish: one more Nelder-Mead from best solution
    if time.time() < deadline - 5:
        x0 = best_points.flatten()
        try:
            result = minimize(_soft_min_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 
                                      'adaptive': True, 'maxfev': 10000})
            
            candidate_points = result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
