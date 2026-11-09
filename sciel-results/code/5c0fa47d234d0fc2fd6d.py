import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2

# Precompute indices for vectorized area calculation
# i < j < k
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def get_areas(points):
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross)

def min_triangle_area_vec(points):
    return np.min(get_areas(points))

def is_inside(points):
    x = points[:, 0]
    y = points[:, 1]
    if np.any(y < -1e-9):
        return False
    if np.any(y > np.sqrt(3) * x + 1e-9):
        return False
    if np.any(y > np.sqrt(3) * (1 - x) + 1e-9):
        return False
    return True

def expand_symmetric(params, template_type):
    """
    Expand reduced parameters to full 11 points.
    template_type 0: 1 center, 5 pairs
    template_type 1: 3 center, 4 pairs
    """
    points = np.zeros((N, 2))
    
    if template_type == 0:
        # 1 center point: (0.5, y0)
        points[0] = [0.5, params[0]]
        
        # 5 pairs: (x_i, y_i) and (1-x_i, y_i)
        for i in range(5):
            x = params[1 + 2*i]
            y = params[2 + 2*i]
            idx1 = 1 + 2*i
            idx2 = 2 + 2*i
            points[idx1] = [x, y]
            points[idx2] = [1.0 - x, y]
            
    elif template_type == 1:
        # 3 center points: (0.5, y0), (0.5, y1), (0.5, y2)
        points[0] = [0.5, params[0]]
        points[1] = [0.5, params[1]]
        points[2] = [0.5, params[2]]
        
        # 4 pairs: (x_i, y_i) and (1-x_i, y_i)
        for i in range(4):
            x = params[3 + 2*i]
            y = params[4 + 2*i]
            idx1 = 3 + 2*i
            idx2 = 4 + 2*i
            points[idx1] = [x, y]
            points[idx2] = [1.0 - x, y]
            
    else:
        # Generic fallback: just take first 11*2 params? No, we only call with 0 or 1.
        pass
        
    return points

def objective_smooth(params, template_type, beta=50.0):
    """
    Smooth objective for DE. Returns negative soft-min area.
    """
    points = expand_symmetric(params, template_type)
    
    # Check validity
    if not is_inside(points):
        return 1.0
        
    areas = get_areas(points)
    
    # Soft min: -1/beta * log(sum(exp(-beta * area)))
    # To avoid overflow, subtract max
    # min_area is approx -1/beta * log(sum(exp(-beta*(area - min_area)))) + min_area
    # But we want to maximize min area, so minimize -min area.
    # Let's just compute the soft min value.
    min_a = np.min(areas)
    exp_terms = np.exp(-beta * (areas - min_a))
    soft_min = min_a - (1.0 / beta) * np.log(np.sum(exp_terms))
    
    return -soft_min

def objective_exact(params, template_type):
    """
    Exact objective for local refinement. Returns negative min area.
    """
    points = expand_symmetric(params, template_type)
    if not is_inside(points):
        return 1.0
    return -min_triangle_area_vec(points)

def refine_local_reduced(params, template_type, deadline):
    """
    Refine in the reduced parameter space using Nelder-Mead with exact objective.
    """
    best_params = params.copy()
    best_score = -objective_exact(best_params, template_type)
    
    rng = np.random.default_rng(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            x0 = best_params.copy()
        else:
            scale = 0.01 * (attempt + 1)
            x0 = best_params + rng.normal(0, scale, size=len(best_params))
            # Basic bounds check for reduced params to avoid crazy values
            # x in [0, 1], y in [0, H]
            # Center points y in [0, H]
            # Pair points x in [0, 1], y in [0, H]
            # We'll just let the objective function penalize invalid points.
            
        try:
            res = minimize(
                lambda p: objective_exact(p, template_type),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if time.time() > deadline:
                break
                
            new_params = res.x
            new_score = -objective_exact(new_params, template_type)
            
            if new_score > best_score:
                best_score = new_score
                best_params = new_params.copy()
                
        except Exception:
            continue
            
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Define templates
    # Template 0: 1 center, 5 pairs. Params: y0, (x1,y1)...(x5,y5) -> 1 + 10 = 11 params
    bounds_t0 = [(0.0, H)] + [(0.0, 1.0), (0.0, H)] * 5
    # Template 1: 3 center, 4 pairs. Params: y0,y1,y2, (x1,y1)...(x4,y4) -> 3 + 8 = 11 params
    bounds_t1 = [(0.0, H)] * 3 + [(0.0, 1.0), (0.0, H)] * 4
    
    templates = [
        (0, bounds_t0),
        (1, bounds_t1)
    ]
    
    for t_idx, bounds in templates:
        if time.time() > deadline:
            break
            
        try:
            # Phase 1: Global Search with DE using smooth objective
            result = differential_evolution(
                lambda p: objective_smooth(p, t_idx, beta=50.0),
                bounds,
                maxiter=100, 
                popsize=15,
                tol=0.01,
                seed=42 + t_idx,
                polish=False,
                init='sobol'
            )
            
            best_params = result.x
            candidate_points = expand_symmetric(best_params, t_idx)
            
            if is_inside(candidate_points):
                candidate_score = min_triangle_area_vec(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
                    best_params = best_params.copy()
                    
        except Exception:
            continue
            
    # If DE didn't find anything, fallback
    if best_points is None:
        # Random valid points
        rng = np.random.default_rng(42)
        for _ in range(100):
            pts = rng.random((N, 2))
            pts[:, 0] *= 1.0
            pts[:, 1] *= H
            if is_inside(pts):
                best_points = pts
                best_score = min_triangle_area_vec(best_points)
                break
        if best_points is None:
            # Grid fallback
            best_points = np.zeros((N, 2))
            for i in range(N):
                best_points[i, 0] = 0.1 + 0.8 * (i / (N-1))
                best_points[i, 1] = 0.1 + 0.6 * ((i % 3) / 2.0)
            # Ensure inside
            if not is_inside(best_points):
                best_points = np.array([[0.5, 0.1] * N]).reshape(N, 2)
            best_score = min_triangle_area_vec(best_points)
            
    # Phase 2: Local Refinement
    # We need to map best_points back to reduced params if we want to refine in reduced space.
    # But we lost the mapping. 
    # Strategy: 
    # 1. If we have best_params from the successful DE run, refine those.
    # 2. Otherwise, try to fit best_points to a template? Hard.
    # Better: Keep track of which template produced the best result.
    
    # Let's restructure to keep track of the best template's params.
    
    # Re-doing the logic to keep params
    
    best_params_global = None
    best_template_global = -1
    
    for t_idx, bounds in templates:
        if time.time() > deadline:
            break
            
        try:
            result = differential_evolution(
                lambda p: objective_smooth(p, t_idx, beta=50.0),
                bounds,
                maxiter=100, 
                popsize=15,
                tol=0.01,
                seed=42 + t_idx,
                polish=False,
                init='sobol'
            )
            
            params_candidate = result.x
            points_candidate = expand_symmetric(params_candidate, t_idx)
            
            if is_inside(points_candidate):
                score_candidate = min_triangle_area_vec(points_candidate)
                if score_candidate > best_score:
                    best_score = score_candidate
                    best_points = points_candidate.copy()
                    best_params_global = params_candidate.copy()
                    best_template_global = t_idx
                    
        except Exception:
            continue
            
    # Refine the best candidate found in reduced space
    if best_params_global is not None and time.time() < deadline - 10:
        refined_params, refined_score = refine_local_reduced(best_params_global, best_template_global, deadline)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = expand_symmetric(refined_params, best_template_global)
            
    # Final Polish: Full space Nelder-Mead on the best points found
    if time.time() < deadline - 5:
        # Flatten points
        x0 = best_points.flatten()
        
        def objective_full(params):
            pts = params.reshape(N, 2)
            if not is_inside(pts):
                return 1.0
            return -min_triangle_area_vec(pts)
            
        try:
            res = minimize(
                objective_full,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            
            if time.time() < deadline:
                new_points = res.x.reshape(N, 2)
                if is_inside(new_points):
                    new_score = min_triangle_area_vec(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        
        except Exception:
            pass
            
    return best_points
