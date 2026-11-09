import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time

N = 14

def _triangle_areas(points):
    """
    Compute areas of all triangles formed by triples of points.
    points: (N, 2) array
    Returns: array of shape (M,) where M = N*(N-1)*(N-2)/6
    """
    n = points.shape[0]
    # Get indices for all triples i < j < k
    idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    i, j, k = idx[:, mask]
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    # Area = 0.5 * | (pj - pi) x (pk - pi) |
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
        
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        
        areas = _triangle_areas(points)
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def _soft_min(values, alpha):
    """
    Soft minimum approximation.
    """
    # Handle potential overflow by shifting
    v = np.asarray(values)
    min_v = np.min(v)
    exp_sum = np.sum(np.exp(-alpha * (v - min_v)))
    if exp_sum == 0:
        return min_v
    return min_v - np.log(exp_sum) / alpha

def _objective_soft(points_flat, alpha):
    """
    Objective function for minimization using soft-min.
    Minimize -soft_score.
    """
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    areas = _triangle_areas(points)
    sm = _soft_min(areas, alpha)
    
    # We want to maximize sm / hull_area
    # So we minimize -sm / hull_area
    return -(sm / hull_area)

def _objective_hard(points_flat):
    """
    Objective function for final polishing using hard min.
    """
    points = points_flat.reshape(N, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    rng = np.random.default_rng(42)
    
    # List of starting points for local optimization
    starts = [best_points.copy()]
    
    # Add some random perturbations of the regular polygon
    for _ in range(5):
        perturbed = init_points + rng.normal(0, 0.1, (N, 2))
        starts.append(perturbed)
        
    # Add random points in the unit disk
    for _ in range(5):
        r = np.sqrt(rng.random(N))
        theta = 2 * np.pi * rng.random(N)
        p = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        starts.append(p)

    # Optimization Loop
    for start_idx, start_point in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        current_points = start_point.copy()
        
        # Phase 1: Soft-min optimization with L-BFGS-B
        # Anneal alpha from low to high
        alphas = [10.0, 50.0, 200.0, 1000.0]
        
        for alpha in alphas:
            if time.time() > deadline - 5:
                break
                
            try:
                res = minimize(
                    _objective_soft, 
                    current_points.flatten(), 
                    args=(alpha,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_points = res.x.reshape(N, 2)
            except Exception:
                pass
        
        # Phase 2: Hard-min optimization with Nelder-Mead for final polish
        if time.time() > deadline - 2:
            break
            
        try:
            res = minimize(
                _objective_hard, 
                current_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            current_points = res.x.reshape(N, 2)
        except Exception:
            pass
        
        current_score = _score(current_points)
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
    # If we still have time, try a quick global search with Differential Evolution
    # This is risky for time, so we only do it if we have > 20s left
    if time.time() < deadline - 20:
        try:
            bounds = [(-1.5, 1.5)] * (2 * N)
            de_res = differential_evolution(
                _objective_hard, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=10,
                tol=1e-6,
                init='sobol'
            )
            de_points = de_res.x.reshape(N, 2)
            de_score = _score(de_points)
            
            if de_score > best_score:
                best_score = de_score
                best_points = de_points.copy()
                
                # Polish the DE result
                if time.time() < deadline - 5:
                    res = minimize(
                        _objective_hard, 
                        best_points.flatten(), 
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    best_points = res.x.reshape(N, 2)
                    best_score = _score(best_points)
        except Exception:
            pass

    return best_points
