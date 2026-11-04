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

def _coordinate_polish(points, time_limit):
    """
    Coordinate-level polish: optimize each point's x and y independently.
    Uses soft-min for smooth gradients.
    """
    start_time = time.time()
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    # Two-stage beta: start broad, then sharpen
    betas = [50.0, 250.0]
    
    for beta in betas:
        if time.time() > start_time + time_limit * 0.8:
            break
        
        for iteration in range(3):
            if time.time() > start_time + time_limit * 0.8:
                break
            
            improved = False
            for i in range(N):
                if time.time() > start_time + time_limit * 0.8:
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
                        options={'xatol': 1e-8, 'maxiter': 100}
                    )
                    if -res.fun > best_score:
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
                        options={'xatol': 1e-8, 'maxiter': 100}
                    )
                    if -res.fun > best_score:
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
            options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000, 'adaptive': True}
        )
        
        if -res.fun > best_score:
            best_score = -res.fun
            best_points = res.x.reshape(-1, 2)
    except Exception:
        pass
    
    # Try coordinate polish with soft-min for better gradient flow
    if time.time() - start_time < time_limit * 0.5:
        coord_time = min(15.0, time_limit * 0.4, deadline_global - time.time() - 5.0)
        if coord_time > 5.0:
            best_points, best_score = _coordinate_polish(best_points, coord_time)
    
    # Try Powell if time permits
    if time.time() - start_time < time_limit * 0.5:
        try:
            res = minimize(
                lambda p: -_calculate_score(p.reshape(-1, 2)),
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 500, 'ftol': 1e-8, 'xtol': 1e-8}
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
        # Increase popsize and maxiter for better exploration
        popsize = 20
        maxiter = 100
        
        res = differential_evolution(
            lambda p: -_score_ring_params(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
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
                # Use L-BFGS-B with soft objective
                beta = 100.0
                res_lbfgs = minimize(
                    lambda p: -_soft_score_ring_params(n_out, n_in, p, beta),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                # Evaluate the exact score at the L-BFGS-B result
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
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000}
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
    
    candidates = [
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
    
    best_overall_score = 0.0
    best_overall_points = None
    
    for n_out, n_in in candidates:
        if time.time() > deadline_global - 10: # Reserve 10s for polishing and final assembly
            break
            
        # Time budget for this candidate
        if (n_out, n_in) == (7, 7):
            t_limit = 80.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 30.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 15.0
        else:
            t_limit = 5.0
            
        score, params = _optimize_ring(n_out, n_in, t_limit)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r, theta = params
                # Generate points
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
        polish_time = min(25.0, deadline_global - time.time() - 2.0)
        if polish_time > 5.0:
            best_overall_points, best_overall_score = _polish_points(best_overall_points, polish_time)
            
    return best_overall_points
