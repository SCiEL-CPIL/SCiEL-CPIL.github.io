import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area computation
    n = points.shape[0]
    total_triangles = n * (n - 1) * (n - 2) // 6
    areas = np.empty(total_triangles)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            v1 = points[j] - points[i]
            v2 = points[j + 1:] - points[i]
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            count = len(cross)
            areas[idx:idx + count] = 0.5 * np.abs(cross)
            idx += count
    
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """Smooth surrogate for min triangle area / hull area using log-sum-exp."""
    if points.shape != (N, 2):
        return 1e6
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e6
    except Exception:
        return 1e6

    n = points.shape[0]
    total_triangles = n * (n - 1) * (n - 2) // 6
    areas = np.empty(total_triangles)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            v1 = points[j] - points[i]
            v2 = points[j + 1:] - points[i]
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            count = len(cross)
            areas[idx:idx + count] = 0.5 * np.abs(cross)
            idx += count
    
    # Normalize areas by hull area
    norm_areas = areas / hull_area
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta * a))) / beta
    # This approximates min(a) as beta -> infinity
    max_val = np.max(norm_areas)
    log_sum = np.log(np.sum(np.exp(-beta * (norm_areas - max_val)))) + beta * max_val
    soft_min = -log_sum / beta
    
    return -soft_min  # Negative for minimization

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    # Outer ring: n_out points at radius 1.0
    # Inner ring: n_in points at radius r_in
    # Angles aligned with offset phi_rel
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + phi_rel
    
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
    
    return np.vstack((pts_out, pts_in))

def _optimize_two_ring(n_out, n_in, time_limit=10.0):
    start_time = time.time()
    
    def objective_soft(params):
        r_in, phi_rel = params
        if r_in < 0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        return _soft_min_score(pts, beta=50.0)

    def objective_exact(params):
        r_in, phi_rel = params
        if r_in < 0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        score = _calculate_score(pts)
        return -score

    # Phase 1: Coarse search with soft-min surrogate
    best_score = 0
    best_params = None
    
    initial_guesses = []
    for r0 in [0.3, 0.5, 0.7, 0.85, 1.0]:
        for phi0 in [0, np.pi / n_in, np.pi / n_out, 2 * np.pi / n_in, 2 * np.pi / n_out]:
            initial_guesses.append([r0, phi0])
    
    for x0 in initial_guesses:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            res = minimize(objective_soft, x0, method='Nelder-Mead', 
                          options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 300})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Phase 2: Refine with exact objective
    if best_params is not None:
        try:
            res = minimize(objective_exact, best_params, method='Nelder-Mead',
                          options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 500})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        return 0, None
        
    return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Test various splits, prioritizing (7,7) which is most promising
    splits = [
        (7, 7),
        (8, 6),
        (12, 2),
        (11, 3),
        (10, 4),
        (9, 5),
        (13, 1),
        (14, 0)
    ]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 15:
            break
        
        if n_in == 0:
            pts = _generate_two_ring(n_out, 0, 0, 0)
            score = _calculate_score(pts)
            try_update(score, pts)
            continue
            
        # Allocate more time to (7,7) configuration
        if n_out == 7 and n_in == 7:
            tl = min(30.0, deadline - time.time() - 15)
        else:
            tl = min(10.0, deadline - time.time() - 15)
        
        if tl <= 1.0:
            break
            
        score, params = _optimize_two_ring(n_out, n_in, time_limit=tl)
        if params is not None:
            pts = _generate_two_ring(n_out, n_in, params[0], params[1])
            
            # Polish with full coordinate optimization using soft-min first, then exact
            def neg_score_flat_soft(x):
                p = x.reshape(N, 2)
                return _soft_min_score(p, beta=100.0)
            
            def neg_score_flat_exact(x):
                p = x.reshape(N, 2)
                return -_calculate_score(p)
            
            # Soft-min polish
            try:
                if time.time() < deadline - 10:
                    res = minimize(neg_score_flat_soft, pts.flatten(), method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-10})
                    new_pts = res.x.reshape(N, 2)
                    new_score = _calculate_score(new_pts)
                    if new_score > score:
                        score = new_score
                        pts = new_pts
            except Exception:
                pass
            
            # Exact polish
            try:
                if time.time() < deadline - 5:
                    res = minimize(neg_score_flat_exact, pts.flatten(), method='L-BFGS-B',
                                  options={'maxiter': 300, 'ftol': 1e-12})
                    new_pts = res.x.reshape(N, 2)
                    new_score = _calculate_score(new_pts)
                    if new_score > score:
                        score = new_score
                        pts = new_pts
            except Exception:
                pass
            
            try_update(score, pts)

    # Final polish on the best solution found
    if best_points is not None and time.time() < deadline - 5:
        def neg_score_final(x):
            p = x.reshape(N, 2)
            return -_calculate_score(p)
        
        try:
            res = minimize(neg_score_final, best_points.flatten(), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14})
            final_pts = res.x.reshape(N, 2)
            final_score = _calculate_score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts
        except Exception:
            pass

    if best_points is None:
        # Fallback: Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
