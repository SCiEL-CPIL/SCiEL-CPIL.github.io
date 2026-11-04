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
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate for min triangle area / hull area using log-sum-exp.
    Returns soft_min_area / hull_area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # Stabilized: m - 1/beta * log(sum(exp(-beta * (areas - m))))
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

def _soft_score_ring_params(n_out, n_in, params, beta=50.0):
    """
    Soft objective function for gradient-based optimization.
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return _soft_min_score(points, beta)

def _optimize_ring(n_out, n_in, time_limit, deadline):
    """
    Optimize a specific ring configuration.
    """
    start_time = time.time()
    
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Global search with Differential Evolution
    try:
        # Use a moderate popsize and maxiter to find a good starting point
        # If time limit is large, use more iterations
        if time_limit > 30:
            popsize = 15
            maxiter = 50
        elif time_limit > 15:
            popsize = 12
            maxiter = 40
        else:
            popsize = 10
            maxiter = 30
            
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
    # This helps navigate the non-smooth landscape by using a smooth surrogate
    if time.time() < deadline - 5.0:
        starts = []
        
        # Add DE result
        if best_params is not None:
            starts.append(best_params)
            
        # Add heuristic starts
        for r_start in [0.3, 0.5, 0.7, 0.9]:
            for theta_start in [0.0, theta_max/2, theta_max/3, 2*theta_max/3]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() > deadline - 5.0:
                break
            try:
                # Use L-BFGS-B with soft objective
                beta = 50.0
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
    if time.time() < deadline - 5.0 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_score_ring_params(n_out, n_in, p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _coordinate_polish(points, deadline):
    """
    Polish all coordinates directly using L-BFGS-B with soft-min surrogate.
    This can escape local optima of the ring parameterization.
    """
    if time.time() > deadline - 5.0:
        return points, _calculate_score(points)
    
    best_points = points.copy()
    best_score = _calculate_score(points)
    
    # Use soft-min for smooth gradient
    def neg_soft_score(x):
        pts = x.reshape(-1, 2)
        return _soft_min_score(pts, beta=50.0)
    
    try:
        # Try multiple betas? Or just one good one.
        # L-BFGS-B on 28 variables.
        res = minimize(
            neg_soft_score,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        polished = res.x.reshape(-1, 2)
        polished_score = _calculate_score(polished)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    except Exception:
        pass
    
    # If we still have time, do a Nelder-Mead polish on the coordinates?
    # That might be too slow for 28 dimensions.
    # But we can try if time permits.
    if time.time() < deadline - 5.0:
        try:
            res_nm = minimize(
                lambda x: -_calculate_score(x.reshape(-1, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
            )
            polished = res_nm.x.reshape(-1, 2)
            polished_score = _calculate_score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
        except Exception:
            pass
            
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin (150s limit)
    
    # Candidate configurations (n_out, n_in)
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
        if time.time() > deadline - 10: # Reserve 10s for final polish
            break
            
        # Time budget for this candidate
        if (n_out, n_in) == (7, 7):
            t_limit = 60.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 25.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 15.0
        else:
            t_limit = 5.0
            
        score, params = _optimize_ring(n_out, n_in, t_limit, deadline)
        
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
        
    # Final Coordinate Polish
    # Use remaining time to polish the best candidate found so far
    if time.time() < deadline - 5.0:
        best_overall_points, best_overall_score = _coordinate_polish(best_overall_points, deadline)
        
    return best_overall_points
