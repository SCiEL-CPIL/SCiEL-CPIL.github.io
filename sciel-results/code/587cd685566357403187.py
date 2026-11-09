import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

def _get_all_triangle_areas(points):
    """Calculate areas of all C(N,3) triangles formed by the points."""
    n = len(points)
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

def _normalize_points(points):
    """Center and normalize to unit hull area."""
    points = points - np.mean(points, axis=0)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return points
    return points / np.sqrt(hull_area)

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
        return 1e6
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate score
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

def _parametrized_points(params):
    """
    Convert parametrized representation to points.
    params: array of shape (N,) representing angular perturbations
            from a regular N-gon base.
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles + params
    
    # Convert to Cartesian
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack([x, y])
    
    return points

def _parametrized_objective(params):
    """
    Objective for parametrized optimization.
    """
    points = _parametrized_points(params)
    points = _normalize_points(points)
    min_area = _min_triangle_area(points)
    return -min_area

def _parametrized_smooth_objective(params, tau=100.0):
    """
    Smooth surrogate for parametrized optimization.
    """
    points = _parametrized_points(params)
    points = _normalize_points(points)
    areas = _get_all_triangle_areas(points)
    
    m = np.min(areas)
    shifted = areas - m
    sum_exp = np.sum(np.exp(-tau * shifted))
    smooth_val = -m + (1.0 / tau) * np.log(sum_exp)
    
    return smooth_val

def _two_ring_points(params):
    """
    Two-ring parametrization: 7 outer, 7 inner points.
    params: [r_inner, angle_offset, *7 outer angle perturbations, *7 inner angle perturbations]
    """
    r_inner = params[0]
    angle_offset = params[1]
    outer_pert = params[2:9]
    inner_pert = params[9:16]
    
    base_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_angles = base_angles + outer_pert
    inner_angles = base_angles + angle_offset + inner_pert
    
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = r_inner * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    points = np.vstack([outer, inner])
    return points

def _two_ring_objective(params):
    """
    Objective for two-ring optimization.
    """
    points = _two_ring_points(params)
    points = _normalize_points(points)
    min_area = _min_triangle_area(points)
    return -min_area

def _two_ring_smooth_objective(params, tau=100.0):
    """
    Smooth surrogate for two-ring optimization.
    """
    points = _two_ring_points(params)
    points = _normalize_points(points)
    areas = _get_all_triangle_areas(points)
    
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
        for tri_idx in min_indices[:10]:
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
    
    def _update_best(points):
        nonlocal best_score, best_points
        if points is not None:
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
    
    # === Strategy 1: Two-ring differential evolution ===
    if time.time() < deadline - 20:
        try:
            # params: [r_inner, angle_offset, 7 outer perturbations, 7 inner perturbations]
            bounds = [(0.3, 0.8), (-0.5, 0.5)] + [(-0.3, 0.3)] * 14
            
            res_de = differential_evolution(
                _two_ring_objective,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                polish=True,
                init='sobol'
            )
            
            if time.time() < deadline - 10:
                p_two_ring = _two_ring_points(res_de.x)
                p_two_ring = _normalize_points(p_two_ring)
                _update_best(p_two_ring)
                
                # Polish with Nelder-Mead in Cartesian space
                if time.time() < deadline - 5:
                    res_nm = minimize(_neg_objective, p_two_ring.flatten(), 
                                      method='Nelder-Mead', 
                                      options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                    p_polished = res_nm.x.reshape(-1, 2)
                    p_polished = _normalize_points(p_polished)
                    _update_best(p_polished)
                    
                    # Local refinement
                    if time.time() < deadline - 2:
                        p_refined, s_ref = _local_refine(p_polished, max_iters=50, step_size=0.003)
                        _update_best(p_refined)
        except Exception:
            pass
    
    # === Strategy 2: Angular perturbation from regular 14-gon ===
    if time.time() < deadline - 15:
        try:
            # params: N angular perturbations
            bounds = [(-0.4, 0.4)] * N
            
            res_de2 = differential_evolution(
                _parametrized_objective,
                bounds=bounds,
                seed=123,
                maxiter=150,
                popsize=12,
                tol=1e-6,
                polish=True,
                init='sobol'
            )
            
            if time.time() < deadline - 5:
                p_angular = _parametrized_points(res_de2.x)
                p_angular = _normalize_points(p_angular)
                _update_best(p_angular)
                
                # Polish
                if time.time() < deadline - 3:
                    res_nm2 = minimize(_neg_objective, p_angular.flatten(), 
                                       method='Nelder-Mead', 
                                       options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                    p_polished2 = res_nm2.x.reshape(-1, 2)
                    p_polished2 = _normalize_points(p_polished2)
                    _update_best(p_polished2)
        except Exception:
            pass
    
    # === Strategy 3: Standard multi-start with smooth + exact ===
    configs = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(x0_reg)
    
    # Two-ring: 7 outer, 7 inner
    ang1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi/7
    inner = 0.55 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    configs.append(np.vstack([outer, inner]))
    
    # Two-ring with different radius
    inner2 = 0.45 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    configs.append(np.vstack([outer, inner2]))
    
    # Random
    for _ in range(3):
        configs.append(np.random.randn(N, 2))
    
    for x_init in configs:
        if time.time() > deadline - 5:
            break
        
        try:
            # Phase 1: Smooth optimization
            res_smooth = minimize(lambda x: _smooth_neg_min_area(x, tau=100.0), 
                                  x_init.flatten(), 
                                  method='Powell', 
                                  options={'maxiter': 400, 'xtol': 1e-7, 'ftol': 1e-9, 'adaptive': True})
            
            if time.time() > deadline - 3:
                break
                
            p_opt = res_smooth.x.reshape(-1, 2)
            p_opt = _normalize_points(p_opt)
            
            # Phase 2: Exact polish
            res_exact = minimize(_neg_objective, 
                                 p_opt.flatten(), 
                                 method='Nelder-Mead', 
                                 options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            
            if time.time() > deadline - 2:
                break
                
            p_polished = res_exact.x.reshape(-1, 2)
            p_polished = _normalize_points(p_polished)
            
            # Phase 3: Local refinement
            if time.time() < deadline - 1:
                p_polished, s_local = _local_refine(p_polished, max_iters=30, step_size=0.005)
            
            _update_best(p_polished)
        except Exception:
            pass
    
    # === Final polish on best solution ===
    if best_points is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(_neg_objective, 
                                 best_points.flatten(), 
                                 method='Nelder-Mead', 
                                 options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
            p_final = res_final.x.reshape(-1, 2)
            p_final = _normalize_points(p_final)
            _update_best(p_final)
        except Exception:
            pass
    
    if best_points is None:
        best_points = x0_reg.copy()
        best_points = _normalize_points(best_points)

    return best_points
