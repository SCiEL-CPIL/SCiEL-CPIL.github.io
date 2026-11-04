import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _compute_cross(points):
    """Compute cross products for all triples."""
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    return cross

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate Minimum Triangle Area
    cross = _compute_cross(points)
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta=100.0):
    """
    Smooth surrogate for min triangle area / hull area using log-sum-exp.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    cross = _compute_cross(points)
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    sum_exp = np.sum(exp_terms)
    soft_min_area = max_area - np.log(sum_exp) / beta
    
    return soft_min_area / hull_area

def _score_ring_params(n_out, n_in, params):
    """
    Objective function for differential evolution.
    params: [r, theta]
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    # Generate outer points
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    # Generate inner points
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return _calculate_score(points)

def _soft_score_ring_params(n_out, n_in, params, beta=100.0):
    """
    Soft objective function for gradient-based optimization.
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    # Generate outer points
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    # Generate inner points
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return _soft_min_score(points, beta)

def _score_three_ring_params(n1, n2, n3, params):
    """
    Objective for three-ring configuration.
    params: [r2, r3, phi2, phi3]
    """
    r2, r3, phi2, phi3 = params
    if r2 < 0 or r2 > 0.999 or r3 < 0 or r3 > 0.999:
        return 0.0
    
    # Outer ring (unit circle)
    angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    pts1 = np.column_stack((np.cos(angles1), np.sin(angles1)))
    
    # Middle ring
    angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + phi2
    pts2 = r2 * np.column_stack((np.cos(angles2), np.sin(angles2)))
    
    # Inner ring
    angles3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + phi3
    pts3 = r3 * np.column_stack((np.cos(angles3), np.sin(angles3)))
    
    points = np.vstack([pts1, pts2, pts3])
    return _calculate_score(points)

def _soft_score_three_ring_params(n1, n2, n3, params, beta=100.0):
    """
    Soft objective for three-ring configuration.
    """
    r2, r3, phi2, phi3 = params
    if r2 < 0 or r2 > 0.999 or r3 < 0 or r3 > 0.999:
        return 0.0
    
    angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    pts1 = np.column_stack((np.cos(angles1), np.sin(angles1)))
    
    angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + phi2
    pts2 = r2 * np.column_stack((np.cos(angles2), np.sin(angles2)))
    
    angles3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + phi3
    pts3 = r3 * np.column_stack((np.cos(angles3), np.sin(angles3)))
    
    points = np.vstack([pts1, pts2, pts3])
    return _soft_min_score(points, beta)

def _optimize_three_ring(n1, n2, n3, time_limit):
    """
    Optimize a three-ring configuration.
    """
    start_time = time.time()
    
    # Symmetry considerations
    g2 = np.gcd(n1, n2)
    g3 = np.gcd(n1, n3)
    phi2_max = 2 * np.pi / g2
    phi3_max = 2 * np.pi / g3
    
    bounds = [(0.0, 0.999), (0.0, 0.999), (0.0, phi2_max), (0.0, phi3_max)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Global search with DE
    try:
        res = differential_evolution(
            lambda p: -_score_three_ring_params(n1, n2, n3, p),
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-5,
            polish=False
        )
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Phase 2: Local polish with L-BFGS-B on soft objective
    if time.time() - start_time < time_limit * 0.6:
        starts = []
        if best_params is not None:
            starts.append(best_params)
        
        # Heuristic starts
        for r2_s in [0.3, 0.5, 0.7]:
            for r3_s in [0.2, 0.4, 0.6]:
                starts.append([r2_s, r3_s, 0.0, 0.0])
        
        for start in starts:
            if time.time() - start_time > time_limit * 0.8:
                break
            try:
                res_lbfgs = minimize(
                    lambda p: -_soft_score_three_ring_params(n1, n2, n3, p, 100.0),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                exact_score = _score_three_ring_params(n1, n2, n3, res_lbfgs.x)
                if exact_score > best_score:
                    best_score = exact_score
                    best_params = res_lbfgs.x
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead polish on exact objective
    if time.time() - start_time < time_limit * 0.9 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_score_three_ring_params(n1, n2, n3, p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass
    
    if best_params is None:
        return 0.0, None
    
    return best_score, best_params

def _coordinate_polish(points, time_limit):
    """
    Coordinate-level polish: optimize each point's x and y independently.
    Uses soft-min for smooth gradients.
    """
    start_time = time.time()
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    # Multi-stage beta: start broad, then sharpen
    betas = [30.0, 100.0, 300.0]
    
    for beta in betas:
        if time.time() > start_time + time_limit * 0.85:
            break
        
        for iteration in range(2):
            if time.time() > start_time + time_limit * 0.85:
                break
            
            improved = False
            for i in range(N):
                if time.time() > start_time + time_limit * 0.85:
                    break
                
                # Optimize x[i]
                def obj_x(x_val):
                    pts = best_points.copy()
                    pts[i, 0] = x_val
                    return -_soft_min_score(pts, beta)
                
                try:
                    res = minimize(
                        obj_x,
                        best_points[i, 0],
                        method='Brent',
                        options={'xatol': 1e-9, 'maxiter': 150}
                    )
                    if -res.fun > best_score + 1e-12:
                        best_points[i, 0] = res.x
                        best_score = -res.fun
                        improved = True
                except Exception:
                    pass
                
                # Optimize y[i]
                def obj_y(y_val):
                    pts = best_points.copy()
                    pts[i, 1] = y_val
                    return -_soft_min_score(pts, beta)
                
                try:
                    res = minimize(
                        obj_y,
                        best_points[i, 1],
                        method='Brent',
                        options={'xatol': 1e-9, 'maxiter': 150}
                    )
                    if -res.fun > best_score + 1e-12:
                        best_points[i, 1] = res.x
                        best_score = -res.fun
                        improved = True
                except Exception:
                    pass
            
            if not improved:
                break
    
    # Final exact evaluation
    final_score = _calculate_score(best_points)
    if final_score > best_score:
        best_score = final_score
    
    return best_points, best_score

def _polish_points(points, time_limit):
    """
    Polish the points using local search on the full 28-dimensional space.
    """
    start_time = time.time()
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    # Try Nelder-Mead
    try:
        res = minimize(
            lambda p: -_calculate_score(p.reshape(-1, 2)),
            best_points.flatten(),
            method='Nelder-Mead',
            options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 1500, 'adaptive': True}
        )
        
        if -res.fun > best_score:
            best_score = -res.fun
            best_points = res.x.reshape(-1, 2)
    except Exception:
        pass
    
    # Try coordinate polish with soft-min for better gradient flow
    remaining = time_limit - (time.time() - start_time)
    if remaining > 8.0:
        coord_time = min(12.0, remaining * 0.5, deadline_global - time.time() - 5.0)
        if coord_time > 5.0:
            best_points, best_score = _coordinate_polish(best_points, coord_time)
    
    # Try Powell if time permits
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10.0:
        try:
            res = minimize(
                lambda p: -_calculate_score(p.reshape(-1, 2)),
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 300, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            
            if -res.fun > best_score:
                best_score = -res.fun
                best_points = res.x.reshape(-1, 2)
        except Exception:
            pass
            
    return best_points, best_score

def _optimize_ring(n_out, n_in, time_limit):
    """
    Optimize a specific ring configuration.
    """
    start_time = time.time()
    
    # Symmetry of the system
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Global search with Differential Evolution
    try:
        popsize = 15
        maxiter = 80
        
        res = differential_evolution(
            lambda p: -_score_ring_params(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=False
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Phase 2: Local polish with Soft-Min L-BFGS-B
    if time.time() - start_time < time_limit * 0.6:
        starts = []
        
        # Add DE result
        if best_params is not None:
            starts.append(best_params)
            
        # Add heuristic starts
        for r_start in [0.3, 0.5, 0.7, 0.9]:
            for theta_start in [0.0, theta_max/2, theta_max/3, 2*theta_max/3]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() - start_time > time_limit * 0.8:
                break
            try:
                beta = 100.0
                res_lbfgs = minimize(
                    lambda p: -_soft_score_ring_params(n_out, n_in, p, beta),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                exact_score = _score_ring_params(n_out, n_in, res_lbfgs.x)
                if exact_score > best_score:
                    best_score = exact_score
                    best_params = res_lbfgs.x
            except Exception:
                pass
                
    # Phase 3: High-precision polish with Nelder-Mead on exact objective
    if time.time() - start_time < time_limit * 0.9 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_score_ring_params(n_out, n_in, p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 800}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

deadline_global = None

def construct():
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + 145.0 # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Two-ring candidates
    two_ring_candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9),
        (10, 4),
        (4, 10),
        (11, 3),
        (3, 11),
        (12, 2),
        (2, 12),
        (13, 1),
        (1, 13)
    ]
    
    # Three-ring candidates
    three_ring_candidates = [
        (10, 2, 2),
        (7, 4, 3),
        (8, 3, 3),
        (9, 3, 2),
        (6, 4, 4),
        (7, 3, 4),
        (8, 4, 2),
        (9, 2, 3),
        (10, 3, 1),
        (11, 2, 1),
    ]
    
    # Optimize two-ring configurations
    for n_out, n_in in two_ring_candidates:
        if time.time() > deadline_global - 15:
            break
            
        if (n_out, n_in) == (7, 7):
            t_limit = 50.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 20.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 10.0
        else:
            t_limit = 4.0
            
        score, params = _optimize_ring(n_out, n_in, t_limit)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r, theta = params
                angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
                x_out = np.cos(angles_out)
                y_out = np.sin(angles_out)
                
                angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
                x_in = r * np.cos(angles_in)
                y_in = r * np.sin(angles_in)
                
                best_overall_points = np.vstack([
                    np.column_stack((x_out, y_out)),
                    np.column_stack((x_in, y_in))
                ])
                
    # Optimize three-ring configurations
    for n1, n2, n3 in three_ring_candidates:
        if time.time() > deadline_global - 15:
            break
            
        t_limit = 8.0
        score, params = _optimize_three_ring(n1, n2, n3, t_limit)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r2, r3, phi2, phi3 = params
                angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
                pts1 = np.column_stack((np.cos(angles1), np.sin(angles1)))
                
                angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + phi2
                pts2 = r2 * np.column_stack((np.cos(angles2), np.sin(angles2)))
                
                angles3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + phi3
                pts3 = r3 * np.column_stack((np.cos(angles3), np.sin(angles3)))
                
                best_overall_points = np.vstack([pts1, pts2, pts3])
                
    # Fallback if no valid points found
    if best_overall_points is None:
        n_out, n_in = 7, 7
        r, theta = 0.5, 0.0
        angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
        x_out = np.cos(angles_out)
        y_out = np.sin(angles_out)
        angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
        x_in = r * np.cos(angles_in)
        y_in = r * np.sin(angles_in)
        best_overall_points = np.vstack([
            np.column_stack((x_out, y_out)),
            np.column_stack((x_in, y_in))
        ])
        
    # Final Polish: Optimize the points directly
    if time.time() < deadline_global - 5:
        polish_time = min(20.0, deadline_global - time.time() - 2.0)
        if polish_time > 5.0:
            best_overall_points, best_overall_score = _polish_points(best_overall_points, polish_time)
            
    return best_overall_points
