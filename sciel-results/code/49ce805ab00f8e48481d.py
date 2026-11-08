import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _get_all_triangle_areas(points):
    """Calculate areas of all C(N,3) triangles formed by the points."""
    n = len(points)
    # Get all combinations of 3 indices
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    
    v1 = points[j_idx] - points[i_idx]
    v2 = points[k_idx] - points[i_idx]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    areas = _get_all_triangle_areas(points)
    return np.min(areas)

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    
    return min_area / hull_area

def _neg_objective(points_flat):
    """
    Objective function for minimization.
    Normalizes points to unit hull area internally to enforce constraint,
    then returns negative score.
    """
    points = points_flat.reshape(-1, 2)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6 # Penalty for degenerate hull
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate score (which is now just min_area since hull_area is 1)
    min_area = _min_triangle_area(points)
    
    # 4. Return negative for minimization
    return -min_area

def _smooth_neg_min_area(points_flat, tau=100.0):
    """
    Smooth surrogate for the negative min triangle area.
    Uses log-sum-exp approximation.
    """
    points = points_flat.reshape(-1, 2)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate areas using vectorized computation
    areas = _get_all_triangle_areas(points)
    
    # 4. Smooth min using log-sum-exp trick
    m = np.min(areas)
    shifted = areas - m
    sum_exp = np.sum(np.exp(-tau * shifted))
    smooth_val = -m + (1.0 / tau) * np.log(sum_exp)
    
    return smooth_val

def _local_refine(points, max_iters=50, step_size=0.01):
    """
    Local refinement: perturb points that are part of the worst triangles.
    """
    points = points.copy()
    current_score = _score(points)
    
    for _ in range(max_iters):
        areas = _get_all_triangle_areas(points)
        min_area = np.min(areas)
        
        # Find which triangles achieve the minimum
        min_indices = np.where(np.abs(areas - min_area) < 1e-10)[0]
        
        # Get the point indices involved in these triangles
        n = len(points)
        idx = np.arange(n)
        i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
        
        # Get the points involved in the worst triangles
        worst_points = set()
        for tri_idx in min_indices[:10]:  # Limit to first 10 worst triangles
            worst_points.add(i_idx[tri_idx])
            worst_points.add(j_idx[tri_idx])
            worst_points.add(k_idx[tri_idx])
        
        # Try perturbing each worst point
        improved = False
        for pt_idx in list(worst_points):
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = points.copy()
                    new_points[pt_idx] = points[pt_idx] + np.array([dx, dy])
                    new_score = _score(new_points)
                    if new_score > current_score + 1e-12:
                        points = new_points
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    # Generate initial configurations
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(x0_reg)
    
    # 2. Regular 14-gon with noise
    configs.append(x0_reg + 0.05 * np.random.randn(N, 2))
    
    # 3. Two rings: 7 outer, 7 inner
    ang1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi/7
    inner = 0.5 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    configs.append(np.vstack([outer, inner]))
    
    # 4. Three rings: 5, 5, 4
    ang1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    ring2 = 0.6 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    ang3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    ring3 = 0.3 * np.column_stack([np.cos(ang3), np.sin(ang3)])
    configs.append(np.vstack([ring1, ring2, ring3]))
    
    # 5. Random points
    configs.append(np.random.randn(N, 2))
    
    # Add more diverse random starts
    for _ in range(5):
        configs.append(np.random.randn(N, 2))

    candidates = []

    for x_init in configs:
        if time.time() > deadline:
            break
            
        try:
            # Phase 1: Smooth optimization with Powell
            # Powell is robust and doesn't need gradients
            res_smooth = minimize(lambda x: _smooth_neg_min_area(x, tau=100.0), 
                                  x_init.flatten(), 
                                  method='Powell', 
                                  options={'maxiter': 500, 'xtol': 1e-6, 'ftol': 1e-8, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p_opt = res_smooth.x.reshape(-1, 2)
            
            # Normalize p_opt for consistency
            p_opt = p_opt - np.mean(p_opt, axis=0)
            hull_area = _hull_area(p_opt)
            if hull_area > 1e-10:
                p_opt = p_opt / np.sqrt(hull_area)
            
            # Phase 2: Polish with exact objective using Nelder-Mead
            # Use tight tolerances for final precision
            res_exact = minimize(_neg_objective, 
                                 p_opt.flatten(), 
                                 method='Nelder-Mead', 
                                 options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p_polished = res_exact.x.reshape(-1, 2)
            
            # Normalize final result
            p_polished = p_polished - np.mean(p_polished, axis=0)
            hull_area = _hull_area(p_polished)
            if hull_area > 1e-10:
                p_polished = p_polished / np.sqrt(hull_area)
            
            # Phase 3: Local refinement
            if time.time() < deadline - 10:  # Leave 10s buffer
                p_polished, s_local = _local_refine(p_polished, max_iters=100, step_size=0.005)
                if s_local > _score(p_polished):
                    pass  # Keep the refined version
                s_opt = _score(p_polished)
            else:
                s_opt = _score(p_polished)
            
            if s_opt > best_score:
                best_score = s_opt
                best_points = p_polished.copy()
                
        except Exception as e:
            pass

    if best_points is None:
        # Fallback to regular polygon
        best_points = x0_reg.copy()
        best_points = best_points - np.mean(best_points, axis=0)
        hull_area = _hull_area(best_points)
        if hull_area > 1e-10:
             best_points = best_points / np.sqrt(hull_area)

    # Final polish on the best solution found
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(_neg_objective, 
                                 best_points.flatten(), 
                                 method='Nelder-Mead', 
                                 options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            p_final = res_final.x.reshape(-1, 2)
            p_final = p_final - np.mean(p_final, axis=0)
            hull_area = _hull_area(p_final)
            if hull_area > 1e-10:
                p_final = p_final / np.sqrt(hull_area)
            
            s_final = _score(p_final)
            if s_final > best_score:
                best_points = p_final
        except:
            pass

    return best_points
