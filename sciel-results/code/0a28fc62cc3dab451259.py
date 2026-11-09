import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _get_min_triangle_info(points):
    """
    Returns min_area, list of indices of triangles with area <= min_area + eps,
    and the gradients for those triangles.
    """
    n = len(points)
    min_area = np.inf
    tri_areas = np.zeros((n*(n-1)*(n-2))//6)
    tri_indices = []
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
                area = 0.5 * abs(cross)
                tri_areas[idx] = area
                tri_indices.append((i, j, k, cross))
                if area < min_area:
                    min_area = area
                idx += 1
    
    # Find all triangles that are "close" to the minimum to create a smooth-ish subgradient
    # Using a small epsilon relative to min_area
    eps = min_area * 0.05 + 1e-9
    critical_tris = []
    for t_idx, (i, j, k, cross) in enumerate(tri_indices):
        if tri_areas[t_idx] <= min_area + eps:
            critical_tris.append((i, j, k, cross, tri_areas[t_idx]))
            
    return min_area, critical_tris

def _compute_subgradient(points, critical_tris):
    """
    Compute the subgradient of the minimum triangle area.
    We want to MAXIMIZE the area, so we move in the direction of the gradient.
    """
    grad = np.zeros_like(points)
    n = len(points)
    
    for i, j, k, cross, area in critical_tris:
        if area < 1e-12:
            # Degenerate triangle, push apart randomly or based on neighbors
            # For now, skip or add small noise
            continue
            
        sign = np.sign(cross)
        if sign == 0:
            sign = 1.0
            
        a = points[i]
        b = points[j]
        c = points[k]
        
        # Vectors
        v1 = b - a
        v2 = c - a
        
        # Gradient of 0.5 * |cross| w.r.t points
        # Let cross = v1_x v2_y - v1_y v2_x
        # d(cross)/da = -d(cross)/dv1 - d(cross)/dv2
        # d(cross)/dv1 = (v2_y, -v2_x)
        # d(cross)/dv2 = (-v1_y, v1_x)
        # So d(cross)/da = -(v2_y, -v2_x) - (-v1_y, v1_x) = (-v2_y + v1_y, v2_x - v1_x)
        # d(cross)/db = d(cross)/dv1 = (v2_y, -v2_x)
        # d(cross)/dc = d(cross)/dv2 = (-v1_y, v1_x)
        
        # We want to increase |cross|.
        # If cross > 0, we want to increase cross.
        # If cross < 0, we want to decrease cross (make it more negative).
        # So we multiply the gradient of cross by sign(cross).
        
        factor = 0.5 * sign
        
        # Gradient of area w.r.t a
        grad_a = factor * np.array([-v2[1] + v1[1], v2[0] - v1[0]])
        # Gradient of area w.r.t b
        grad_b = factor * np.array([v2[1], -v2[0]])
        # Gradient of area w.r.t c
        grad_c = factor * np.array([-v1[1], v1[0]])
        
        grad[i] += grad_a
        grad[j] += grad_b
        grad[k] += grad_c
        
    return grad

def _normalize_points(points):
    """
    Normalize points to have centroid at (0,0) and unit area hull.
    This removes translation and scaling degrees of freedom,
    which are irrelevant to the score but can cause drift in optimization.
    """
    # Centroid to 0
    points = points - np.mean(points, axis=0)
    
    # Scale to unit hull area
    hull_area = _hull_area(points)
    if hull_area > 1e-10:
        points = points * np.sqrt(1.0 / hull_area)
    return points

def get_score(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area, _ = _get_min_triangle_info(points)
    return min_area / hull_area

def _subgradient_optimize(init_points, deadline):
    """
    Optimize using subgradient ascent on the minimum triangle area.
    We maximize min_area / hull_area.
    Since we normalize to unit hull area internally during optimization,
    we are effectively maximizing min_area for a fixed hull area.
    """
    points = init_points.copy()
    points = _normalize_points(points)
    
    best_score = get_score(points)
    best_points = points.copy()
    
    step_size = 0.1
    min_step_size = 1e-6
    no_improve_count = 0
    
    while time.time() < deadline:
        min_area, critical_tris = _get_min_triangle_info(points)
        if not critical_tris:
            break
            
        # Compute subgradient of min area
        grad = _compute_subgradient(points, critical_tris)
        
        # Normalize gradient to unit length to control step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad = grad / grad_norm
        
        # Take step
        new_points = points + step_size * grad
        
        # Normalize new points to maintain unit hull area (constraint)
        new_points = _normalize_points(new_points)
        
        # Check for improvement
        new_score = get_score(new_points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            # Optionally increase step size slightly if improving
            # step_size = min(step_size * 1.05, 0.5)
        else:
            no_improve_count += 1
            step_size = max(step_size * 0.5, min_step_size)
            
            if step_size < min_step_size:
                # Stuck in local minimum, try a random perturbation
                rng = np.random.default_rng()
                points = points + rng.normal(0, 0.01, points.shape)
                points = _normalize_points(points)
                step_size = 0.05
                no_improve_count = 0
                
            if no_improve_count > 50:
                break

    return best_points, best_score

def _generate_structured_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular 12-gon + center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    configs.append(np.vstack([pts12, np.array([[0.0, 0.0]])]))
    
    # 3. Two rings: 7 outer, 6 inner
    angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner = np.column_stack([0.55*np.cos(angles6), 0.55*np.sin(angles6)])
    configs.append(np.vstack([outer, inner]))
    
    # 4. 8 outer, 5 inner
    angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer = np.column_stack([np.cos(angles8), np.sin(angles8)])
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    inner = np.column_stack([0.5*np.cos(angles5), 0.5*np.sin(angles5)])
    configs.append(np.vstack([outer, inner]))
    
    # 5. 9 outer, 4 inner
    angles9 = np.linspace(0, 2*np.pi, 9, endpoint=False)
    outer = np.column_stack([np.cos(angles9), np.sin(angles9)])
    angles4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    inner = np.column_stack([0.45*np.cos(angles4), 0.45*np.sin(angles4)])
    configs.append(np.vstack([outer, inner]))
    
    # 6. 10 outer, 3 inner
    angles10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    outer = np.column_stack([np.cos(angles10), np.sin(angles10)])
    angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    inner = np.column_stack([0.4*np.cos(angles3), 0.4*np.sin(angles3)])
    configs.append(np.vstack([outer, inner]))
    
    # 7. 11 outer, 2 inner
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    outer = np.column_stack([np.cos(angles11), np.sin(angles11)])
    inner = np.array([[0.3, 0.0], [-0.3, 0.0]])
    configs.append(np.vstack([outer, inner]))
    
    # 8. 6 outer, 7 inner
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer = np.column_stack([np.cos(angles6), np.sin(angles6)])
    inner6 = np.column_stack([0.4*np.cos(angles6 + np.pi/6), 0.4*np.sin(angles6 + np.pi/6)])
    configs.append(np.vstack([outer, np.array([[0.0, 0.0]]), inner6]))
    
    # 9. Random configurations
    for seed in [42, 123, 777, 2024, 999, 31415]:
        rng = np.random.default_rng(seed)
        pts = rng.random((N, 2)) * 2 - 1
        configs.append(pts)
        
    # 10. Grid-like
    rng = np.random.default_rng(555)
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if len(grid_pts) < N:
                grid_pts.append([i/3.0, j/3.0])
    grid_pts = np.array(grid_pts[:N])
    grid_pts += rng.normal(0, 0.1, grid_pts.shape)
    configs.append(grid_pts)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    configs = _generate_structured_configs()
    
    best_score = -np.inf
    best_points = None
    
    # Phase 1: Subgradient optimization from multiple starts
    for idx, init_points in enumerate(configs):
        if time.time() > deadline - 10.0:
            break
        
        try:
            optimized, score = _subgradient_optimize(init_points, deadline - 10.0)
            if score > best_score:
                best_score = score
                best_points = optimized
        except Exception:
            pass
    
    if best_points is None:
        best_points = configs[0]
        best_score = get_score(best_points)
        
    # Phase 2: Nelder-Mead polish on the best result
    if time.time() < deadline - 5.0:
        try:
            def neg_score(x):
                pts = x.reshape(N, 2)
                return -get_score(pts)
            
            flat = best_points.reshape(-1)
            result = minimize(neg_score, flat, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10,
                                   'adaptive': True})
            new_score = get_score(result.x.reshape(N, 2))
            if new_score > best_score:
                best_score = new_score
                best_points = result.x.reshape(N, 2)
        except Exception:
            pass
            
    return best_points
