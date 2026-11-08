import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 140.0  # Safety margin before 150s kill

def _hull_area(pts):
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas.extend(0.5 * np.abs(cross))
    return np.array(areas)

def _score(pts):
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return 0.0
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    min_a = np.min(areas)
    return min_a / hull_a

def _soft_min(areas, beta):
    """
    Compute the soft minimum of an array of values.
    soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    """
    if len(areas) == 0:
        return 0.0
    # Numerical stability
    min_val = np.min(areas)
    if min_val < 1e-12:
        return 0.0
    # Simpler stable form:
    # soft_min = min_val - (1/beta) * log( sum(exp(-beta*(areas - min_val))) )
    diff = areas - min_val
    exp_terms = np.exp(-beta * diff)
    log_sum = np.log(np.sum(exp_terms))
    return min_val - log_sum / beta

def _objective_soft(pts_flat, beta):
    """
    Objective function to minimize.
    We want to maximize min_area / hull_area.
    Equivalent to minimizing hull_area / min_area.
    Using soft_min: minimize hull_area / soft_min(areas, beta).
    """
    points = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
        
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    sm = _soft_min(areas, beta)
    if sm < 1e-12:
        return 1e10
        
    return hull_area / sm

def _objective_exact(pts_flat):
    """
    Objective function for final polishing.
    Minimizes hull_area / min_area.
    """
    points = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
        
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    min_a = np.min(areas)
    return hull_area / min_a

def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    p_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(p_reg)
    
    # 2. Regular 14-gon with noise
    rng = np.random.RandomState(42)
    for seed in range(5):
        noise = rng.normal(0, 0.1, size=(N, 2))
        p_noisy = p_reg + noise
        configs.append(p_noisy)
        
    # 3. Two concentric rings (7 and 7)
    ang1 = 2 * np.pi * np.arange(7) / 7
    r1 = 0.6
    p1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
    
    ang2 = 2 * np.pi * np.arange(7) / 7 + np.pi/7 # Offset
    r2 = 1.0
    p2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
    
    p_rings = np.vstack([p1, p2])
    configs.append(p_rings)
    
    # 4. Distorted regular polygon
    mod = 1.0 + 0.1 * np.cos(3 * angles)
    p_mod = np.column_stack([mod * np.cos(angles), mod * np.sin(angles)])
    configs.append(p_mod)
    
    # Normalize all to unit area for consistency in initial scoring
    normalized_configs = [_normalize_unit_area(c) for c in configs]
    return normalized_configs

def construct():
    start_time = time.time()
    
    # Get initial configs
    inits = _generate_initial_configs()
    
    # Evaluate initial scores
    best_score = -np.inf
    best_pts = inits[0]
    
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init
            
    # Sort inits by score (best first) to prioritize good starting points
    init_scores = [_score(c) for c in inits]
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    
    # Optimization Loop
    # We will run optimization on the top few initial configurations
    
    # Define betas for annealing
    betas = [50.0, 100.0, 200.0, 400.0]
    
    num_inits_to_try = min(5, len(inits))
    
    for i in range(num_inits_to_try):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        x0 = inits[i].flatten()
        current_best_x = x0.copy()
        current_best_obj = _objective_exact(x0)
        
        # Annealing loop
        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Define objective for this beta
            def obj_func(x, b=beta):
                return _objective_soft(x, b)
                
            try:
                # L-BFGS-B is good for smooth functions
                res = minimize(obj_func, current_best_x, method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.success and res.fun < current_best_obj:
                    current_best_obj = res.fun
                    current_best_x = res.x
            except Exception:
                pass
        
        # Local Polish with Exact Objective (Nelder-Mead)
        if time.time() - start_time < DEADLINE - 2:
            def obj_exact(x):
                return _objective_exact(x)
                
            try:
                res = minimize(obj_exact, current_best_x, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                
                if res.fun < current_best_obj:
                    current_best_obj = res.fun
                    current_best_x = res.x
            except Exception:
                pass
        
        # Check if this result is better than global best
        final_pts = current_best_x.reshape(-1, 2)
        final_score = _score(final_pts)
        
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    # Final Normalization
    # The score is scale invariant, but we should return points in a reasonable range
    # Normalize so hull area is 1? The task says "normalized to unit area".
    # "The notebook normalizes by the point set convex hull area."
    # This implies the evaluator calculates score = min_area / hull_area.
    # So the actual scale of the returned points doesn't matter for the score,
    # but returning them in a standard scale is good practice.
    
    # Let's normalize to unit area hull for the return value.
    final_pts = best_pts
    hull_a = _hull_area(final_pts)
    if hull_a > 1e-10:
        final_pts = final_pts / np.sqrt(hull_a)
        
    return final_pts

if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
