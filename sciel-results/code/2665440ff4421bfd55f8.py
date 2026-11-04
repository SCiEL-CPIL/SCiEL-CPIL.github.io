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
    # points shape (N, 2)
    # TRIPLE_INDICES shape (M, 3)
    
    # Extract points for all triples
    # This creates a (M, 3, 2) array
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    # Compute cross product (pj - pi) x (pk - pi)
    # (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
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
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # To avoid overflow, shift by max
    max_area = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_area))))
    soft_min = max_area - np.log(np.sum(np.exp(-beta * (areas - max_area)))) / beta
    # Wait, the formula for soft min is:
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    # Let's re-derive:
    # min(x) approx -1/beta * log(sum(exp(-beta * x)))
    # To stabilize: let m = max(x).
    # -1/beta * log(exp(-beta*m) * sum(exp(-beta*(x_i - m))))
    # = -1/beta * (-beta*m + log(sum(exp(-beta*(x_i - m)))))
    # = m - 1/beta * log(sum(exp(-beta*(x_i - m))))
    
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
    # Rotate inner points by theta
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
    # Rotate inner points by theta
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return _soft_min_score(points, beta)

def _optimize_ring(n_out, n_in, time_limit):
    """
    Optimize a specific ring configuration.
    """
    start_time = time.time()
    
    # Symmetry of the system
    # The system is invariant under rotation by 2*pi/gcd(n_out, n_in)
    # So we only need to search theta in [0, 2*pi/gcd(n_out, n_in)]
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Global search with Differential Evolution
    try:
        # Use a moderate popsize and maxiter to find a good starting point
        popsize = 10
        maxiter = 30
        
        res = differential_evolution(
            lambda p: -_score_ring_params(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=False # We will do our own polishing
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception as e:
        pass
    
    # Phase 2: Local polish with Soft-Min L-BFGS-B
    # This helps navigate the non-smooth landscape by using a smooth surrogate
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
                # Increase beta gradually for higher precision?
                # Start with moderate beta
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
    if time.time() - start_time < time_limit * 0.9 and best_params is not None:
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin (150s limit)
    
    # Candidate configurations (n_out, n_in)
    # We assume outer points are on the hull.
    # n_out + n_in = 14
    
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
    
    # Allocate time per candidate
    # We have ~140s.
    # Let's give 7-7 more time, others less.
    
    # Sort candidates by likelihood
    # 7-7 is most likely.
    
    for n_out, n_in in candidates:
        if time.time() > deadline - 5: # Reserve 5s for final assembly
            break
            
        # Time budget for this candidate
        # Decrease time for less likely candidates
        if (n_out, n_in) == (7, 7):
            t_limit = 60.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 25.0
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
        # Return a regular 14-gon? Or just the 7-7 default?
        # Let's just return the 7-7 with r=0.5, theta=0
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
        
    return best_overall_points
