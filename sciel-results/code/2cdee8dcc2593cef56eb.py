"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """Calculate areas of all triangles formed by triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the normalized minimum triangle area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri_area = _triangle_areas(points)
    return min_tri_area / hull_area

def _soft_min_score(points, temperature=0.01):
    """
    Smooth approximation of the minimum triangle area using log-sum-exp.
    This is differentiable and suitable for gradient-based optimization.
    """
    n = len(points)
    areas = []
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    
    if len(areas) == 0:
        return 0.0
    
    areas = np.array(areas)
    # Soft minimum using log-sum-exp trick
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0
    scaled = areas / temperature
    log_sum = np.log(np.sum(np.exp(scaled - np.max(scaled)))) + np.max(scaled)
    soft_min = temperature * log_sum - max_area
    
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    return soft_min / hull_area

def _neg_soft_score(points, temperature=0.01):
    """Negative soft-min score for minimization."""
    return -_soft_min_score(points, temperature)

def _neg_score(points):
    """Negative raw score for minimization."""
    return -_score(points)

def _simulated_annealing(points, start_time, deadline):
    """
    Simulated annealing to explore the configuration space.
    Uses soft-min objective for smooth gradient-like behavior.
    """
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Annealing parameters
    T_start = 0.05
    T_end = 0.0001
    cooling_rate = 0.995
    
    T = T_start
    step_size = 0.05
    
    while time.time() < deadline:
        # Generate perturbation
        noise = np.random.normal(0, step_size, (N, 2))
        new_points = current_points + noise
        
        # Evaluate
        new_score = _score(new_points)
        delta = new_score - current_score
        
        # Accept or reject
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        # Cool down
        T *= cooling_rate
        step_size *= 0.999
        
        if T < T_end:
            break
    
    return best_points, best_score

def _local_refinement(points, start_time, deadline):
    """
    Local refinement using Nelder-Mead from a good starting point.
    """
    x0 = points.flatten()
    
    options = {
        'maxiter': 3000,
        'xatol': 1e-9,
        'fatol': 1e-9,
        'adaptive': True
    }
    
    try:
        result = minimize(_neg_score, x0, method='Nelder-Mead', options=options)
        candidate_points = result.x.reshape(N, 2)
        candidate_score = _score(candidate_points)
        return candidate_points, candidate_score
    except Exception:
        return points, _score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Start from regular polygon and anneal
    angles = 2 * np.pi * np.arange(N) / N
    points_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Anneal from regular polygon
    anneal_deadline = min(start_time + 30.0, deadline - 5.0)
    anneal_points, anneal_score = _simulated_annealing(points_reg, start_time, anneal_deadline)
    
    if anneal_score > best_score:
        best_score = anneal_score
        best_points = anneal_points.copy()
    
    # Local refinement from annealed result
    if time.time() < deadline - 5.0:
        refine_deadline = min(start_time + 60.0, deadline - 2.0)
        refine_points, refine_score = _local_refinement(anneal_points, start_time, refine_deadline)
        
        if refine_score > best_score:
            best_score = refine_score
            best_points = refine_points.copy()
    
    # Strategy 2: Try different initial configurations with annealing
    if time.time() < deadline - 10.0:
        # Try a few random initial configurations
        for restart in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # Random initial configuration within unit circle
            np.random.seed(42 + restart)
            angles_rand = np.random.uniform(0, 2 * np.pi, N)
            radii = np.random.uniform(0.5, 1.0, N)
            points_rand = np.column_stack([radii * np.cos(angles_rand), radii * np.sin(angles_rand)])
            
            anneal_deadline = min(start_time + 70.0 + restart * 5.0, deadline - 5.0)
            anneal_points, anneal_score = _simulated_annealing(points_rand, start_time, anneal_deadline)
            
            if anneal_score > best_score:
                best_score = anneal_score
                best_points = anneal_points.copy()
            
            # Quick local refinement
            if time.time() < deadline - 5.0:
                refine_deadline = min(start_time + 80.0 + restart * 5.0, deadline - 2.0)
                refine_points, refine_score = _local_refinement(anneal_points, start_time, refine_deadline)
                
                if refine_score > best_score:
                    best_score = refine_score
                    best_points = refine_points.copy()
    
    # Final polish with Nelder-Mead if time remains
    if time.time() < deadline - 2.0 and best_points is not None:
        final_deadline = deadline - 1.0
        final_points, final_score = _local_refinement(best_points, start_time, final_deadline)
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    if best_points is None:
        # Fallback to regular polygon
        best_points = points_reg
    
    return best_points

# EVOLVE-BLOCK-END
