import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
DEADLINE = 135.0  # Safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

_START_TIME = time.time()

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized for speed.
    """
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    # Iterate over pairs (i, j) and vectorize over k
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # k ranges from j+1 to n-1
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    """
    Compute all triangle areas (vectorized).
    Returns array of shape (C(n,3),) with all triangle areas.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            a = 0.5 * np.abs(cross)
            areas.extend(a)
    return np.array(areas)

def _generate_initial_points():
    """
    Generate diverse initial configurations.
    1. Fekete points (approximated by iterative repulsion)
    2. Regular polygon
    3. Perturbed regular polygon
    4. Random points in unit disk
    """
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts)
    
    # 2. Perturbed regular
    for _ in range(5):
        noise = np.random.uniform(-0.1, 0.1, (N, 2))
        pts_pert = pts + noise
        inits.append(pts_pert)
        
    # 3. Fekete-like points via simple repulsion
    # Start with random points in unit circle
    pts_fek = np.random.uniform(-1, 1, (N, 2))
    norms = np.linalg.norm(pts_fek, axis=1)
    mask = norms > 1.0
    pts_fek[mask] /= norms[mask, None]
    
    # Simple repulsion optimization for Fekete points (maximize min distance)
    # This is a proxy for maximizing area in a bounded region
    def _min_dist_obj(p):
        pts = p.reshape(N, 2)
        # Normalize to unit circle roughly to keep them in bounds
        # Actually, for Fekete we want them in the unit circle.
        # Let's just maximize min distance.
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        return -np.min(dists)
        
    if _time_left() > 100:
        try:
            res = minimize(_min_dist_obj, pts_fek.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            inits.append(res.x.reshape(N, 2))
        except:
            pass
            
    # 4. Two rings (inner 4, outer 10)
    inner_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi/10
    inner_pts = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inits.append(np.vstack((inner_pts, outer_pts)))
    
    # 5. Random
    for _ in range(3):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        pts_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts_rand)
        
    return inits

def _objective(p):
    """
    Objective for SLSQP: Minimize -min_triangle_area / hull_area
    We use a soft-min approach to make it differentiable.
    """
    points = p.reshape(N, 2)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10
        
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 1e10
    
    # Soft min
    beta = 50.0
    min_a = np.min(areas)
    shifted = areas - min_a
    soft_min_val = -1.0/beta * np.log(np.sum(np.exp(-beta * shifted))) + min_a
    
    score = soft_min_val / hull_area
    return -score

def _objective_hard(p):
    """
    Hard objective for final evaluation.
    """
    points = p.reshape(N, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    
    inits = _generate_initial_points()
    
    best_params = inits[0].flatten()
    best_score = _score(inits[0])
    
    # Phase 1: SLSQP on soft objective
    # SLSQP is good for handling the implicit constraints of the geometry
    # via the objective landscape.
    
    for i, init in enumerate(inits):
        if _time_left() < 20:
            break
            
        try:
            # Use SLSQP
            res = minimize(
                _objective,
                init.flatten(),
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.fun < _objective(best_params):
                candidate = res.x.reshape(N, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_params = res.x
                    best_score = cand_score
                    
        except Exception:
            pass
        
        # Phase 2: Nelder-Mead polish on hard objective
        if _time_left() < 15:
            break
            
        try:
            res_nm = minimize(
                _objective_hard,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_nm.fun < _objective_hard(best_params):
                candidate = res_nm.x.reshape(N, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_params = res_nm.x
                    best_score = cand_score
                    
        except Exception:
            pass

    # Final Polish: Powell's method
    if _time_left() > 10:
        try:
            res_pow = minimize(
                _objective_hard,
                best_params,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-14, 'adaptive': True}
            )
            if res_pow.fun < _objective_hard(best_params):
                best_params = res_pow.x
                best_score = _score(res_pow.x.reshape(N, 2))
        except Exception:
            pass

    # Reconstruct and normalize
    points = best_params.reshape(N, 2)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        points = points * scale
        
    return points
