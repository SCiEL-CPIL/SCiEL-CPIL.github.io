"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

def _triangle_areas(points):
    """Compute all triangle areas for triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1e-10

def _soft_min(areas, temperature=0.01):
    """Soft minimum using log-sum-exp for smoothness."""
    # Use negative log-sum-exp for soft min
    neg_areas = -areas
    max_val = np.max(neg_areas)
    if max_val == -np.inf:
        return 0.0
    shifted = neg_areas - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(shifted)))
    return -log_sum_exp

def _objective_flat(flat_points):
    """Objective function: minimize negative of (min_triangle_area / hull_area)."""
    points = flat_points.reshape(N, 2)
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return -(min_area / hull_area)

def _objective_soft(flat_points, temperature=0.05):
    """Objective with soft-min for smoother gradient."""
    points = flat_points.reshape(N, 2)
    areas = _triangle_areas(points)
    soft_min_val = _soft_min(areas, temperature)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return -(soft_min_val / hull_area)

def _initial_config():
    """Generate a good initial configuration: points on a circle."""
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start from circle configuration
    init_points = _initial_config()
    init_flat = init_points.flatten()
    
    best_score = 0.0
    best_points = init_points.copy()
    
    # Evaluate initial
    areas = _triangle_areas(init_points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(init_points)
    if hull_area > 1e-10:
        best_score = min_area / hull_area
    
    # Phase 1: Differential Evolution for global search
    bounds = [(-1, 1)] * (2 * N)
    
    def de_objective(flat):
        if time.time() > deadline - 30:
            return 1e10
        return _objective_flat(flat)
    
    try:
        result_de = differential_evolution(
            de_objective,
            bounds=bounds,
            x0=init_flat,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol'
        )
        
        if result_de.success or result_de.fun < _objective_flat(init_flat):
            de_points = result_de.x.reshape(N, 2)
            areas = _triangle_areas(de_points)
            min_area = np.min(areas)
            hull_area = _convex_hull_area(de_points)
            if hull_area > 1e-10:
                de_score = min_area / hull_area
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        return best_points
    
    # Phase 2: Local polish with L-BFGS-B
    def polish_objective(flat):
        if time.time() > deadline - 5:
            return 1e10
        return _objective_flat(flat)
    
    try:
        result_local = minimize(
            polish_objective,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=[(-1, 1)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        if result_local.fun < _objective_flat(best_points.flatten()):
            local_points = result_local.x.reshape(N, 2)
            areas = _triangle_areas(local_points)
            min_area = np.min(areas)
            hull_area = _convex_hull_area(local_points)
            if hull_area > 1e-10:
                local_score = min_area / hull_area
                if local_score > best_score:
                    best_score = local_score
                    best_points = local_points.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        return best_points
    
    # Phase 3: Try with soft-min objective for potentially smoother landscape
    def soft_objective(flat):
        if time.time() > deadline - 2:
            return 1e10
        return _objective_soft(flat, temperature=0.03)
    
    try:
        result_soft = minimize(
            soft_objective,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=[(-1, 1)] * (2 * N),
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        
        soft_points = result_soft.x.reshape(N, 2)
        areas = _triangle_areas(soft_points)
        min_area = np.min(areas)
        hull_area = _convex_hull_area(soft_points)
        if hull_area > 1e-10:
            soft_score = min_area / hull_area
            if soft_score > best_score:
                best_score = soft_score
                best_points = soft_points.copy()
    except Exception:
        pass
    
    return best_points
# EVOLVE-BLOCK-END
