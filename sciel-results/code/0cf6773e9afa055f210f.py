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

def _soft_score_ring_params(n_out, n_in, params, beta=100.0):
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

def _fourier_polish(points, deadline, n_harmonics=4):
    """
    Polish points using a Fourier radial parameterization.
    This allows asymmetric distortions of the circle that the simple ring
    parameterization cannot capture, potentially escaping local optima.
    
    We parameterize each point as:
    r_i = r_0 * (1 + sum_{k=1}^{K} a_k * cos(k * theta_i + phi_k))
    theta_i = 2 * pi * i / N
    
    Actually, since the points are not necessarily on a circle, we can just
    optimize the radial distances r_i and angles theta_i directly, but with
    a smooth prior or just direct coordinate descent on the exact objective.
    
    However, the inspiration mentions "Fourier-radius 6-dim parameterization".
    Let's try a parameterization where the points are defined by a base radius
    and a small number of Fourier coefficients for the radius as a function of
    angle.
    
    Let points be ordered by angle.
    theta_i = 2*pi*i/N
    r_i = R * (1 + sum_{k=1}^{K} (a_k cos(k*theta_i) + b_k sin(k*theta_i)))
    
    Parameters: R, a_1..a_K, b_1..b_K
    Total params: 1 + 2*K.
    For K=2, that's 5 params.
    For K=3, that's 7 params.
    
    This enforces a smooth shape.
    """
    if time.time() > deadline - 5.0:
        return points, _calculate_score(points)
    
    # Order points by angle to define the sequence
    cx, cy = np.mean(points[:, 0]), np.mean(points[:, 1])
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    sort_idx = np.argsort(angles)
    sorted_points = points[sort_idx]
    
    # Initial parameters from sorted points
    r_0 = np.mean(np.linalg.norm(sorted_points - [cx, cy], axis=1))
    theta_seq = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Least squares fit for Fourier coefficients
    # r_i = R * (1 + sum (a_k cos(k*theta_i) + b_k sin(k*theta_i)))
    # Let's just use direct coordinate optimization on the 28 coords,
    # but maybe initialize with a Fourier fit to ensure smoothness?
    
    # Actually, the current program does `_coordinate_polish`.
    # Let's make `_coordinate_polish` more effective.
    
    # Alternative: Just do a more aggressive coordinate descent.
    # The previous `_coordinate_polish` used L-BFGS-B on soft score.
    # Let's try a few restarts of L-BFGS-B on the soft score with different betas.
    
    best_points = points.copy()
    best_score = _calculate_score(points)
    
    def neg_soft_score(x, beta):
        pts = x.reshape(-1, 2)
        return -_soft_min_score(pts, beta)
    
    # Try a range of betas
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(
                lambda x: neg_soft_score(x, beta),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            polished = res.x.reshape(-1, 2)
            polished_score = _calculate_score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
        except Exception:
            pass
            
    # Final Nelder-Mead on exact score
    if time.time() < deadline - 2.0:
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
    deadline = start_time + 145.0 # Safety margin (150s limit)
    
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
        if time.time() > deadline - 15: # Reserve 15s for final polish
            break
            
        # Time budget for this candidate
        if (n_out, n_in) == (7, 7):
            t_limit = 50.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 20.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 10.0
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
        best_overall_points, best_overall_score = _fourier_polish(best_overall_points, deadline)
        
    return best_overall_points
