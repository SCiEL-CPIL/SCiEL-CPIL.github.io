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
    
    # Phase 1: Global search with Differential Evolution - more iterations
    try:
        popsize = 15
        maxiter = 50
        
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
    
    # Phase 2: Local polish with Soft-Min L-BFGS-B using adaptive beta
    if time.time() - start_time < time_limit * 0.55:
        starts = []
        
        if best_params is not None:
            starts.append(best_params)
            
        # More diverse heuristic starts
        for r_start in [0.2, 0.4, 0.6, 0.8, 0.95]:
            for theta_start in [0.0, theta_max/4, theta_max/2, 3*theta_max/4]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() - start_time > time_limit * 0.75:
                break
            try:
                # Adaptive beta: start lower for exploration, higher for precision
                for beta in [30.0, 80.0, 200.0]:
                    if time.time() - start_time > time_limit * 0.75:
                        break
                    res_lbfgs = minimize(
                        lambda p, b=beta: -_soft_score_ring_params(n_out, n_in, p, b),
                        start,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    
                    exact_score = _score_ring_params(n_out, n_in, res_lbfgs.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_params = res_lbfgs.x
                    start = res_lbfgs.x  # Use result as next start
            except Exception:
                pass
                
    # Phase 3: High-precision polish with Nelder-Mead on exact objective
    if time.time() - start_time < time_limit * 0.95 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_score_ring_params(n_out, n_in, p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1000, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _polish_points(points, time_limit, start_time):
    """
    Directly polish the point coordinates using Nelder-Mead on the exact objective.
    This allows all 28 coordinates to be optimized, not just ring parameters.
    """
    deadline = start_time + time_limit
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    # Nelder-Mead on all 28 coordinates
    try:
        def neg_score(x):
            pts = x.reshape(-1, 2)
            return -_calculate_score(pts)
        
        res = minimize(
            neg_score,
            points.flatten(),
            method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000, 'adaptive': True}
        )
        
        if time.time() < deadline:
            candidate_score = -res.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = res.x.reshape(-1, 2)
    except Exception:
        pass
    
    # Second polish pass with even tighter tolerance
    if time.time() < deadline - 2:
        try:
            def neg_score2(x):
                pts = x.reshape(-1, 2)
                return -_calculate_score(pts)
            
            res2 = minimize(
                neg_score2,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 1000, 'adaptive': True}
            )
            
            candidate_score = -res2.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = res2.x.reshape(-1, 2)
        except Exception:
            pass
    
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin (150s limit)
    
    # Candidate configurations (n_out, n_in)
    # We assume outer points are on the hull.
    # n_out + n_in = 14
    
    # Prioritize configurations most likely to yield high scores
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
            
        # Time budget for this candidate - use more time overall
        if (n_out, n_in) == (7, 7):
            t_limit = 50.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 20.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 12.0
        else:
            t_limit = 5.0
            
        score, params = _optimize_ring(n_out, n_in, t_limit)
        
        if score > best_overall_score and params is not None:
            best_overall_score = score
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
    
    # Final direct polish on all 28 coordinates
    if time.time() < deadline - 3:
        polish_time = deadline - time.time() - 3
        polished_score, polished_points = _polish_points(
            best_overall_points, polish_time, start_time
        )
        if polished_score > best_overall_score:
            best_overall_score = polished_score
            best_overall_points = polished_points
    
    return best_overall_points
