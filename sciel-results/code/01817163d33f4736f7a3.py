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
    
    exp_terms = np.exp(-beta * (areas - np.max(areas)))
    sum_exp = np.sum(exp_terms)
    soft_min_area = np.max(areas) - np.log(sum_exp) / beta
    
    return soft_min_area / hull_area

def _fourier_points(params):
    """
    Generate 14 points on a Fourier-modulated 14-gon.
    params: [a1, b1, a2, b2] or [a1, b1, a2, b2, a3, b3]
    Modes 1 and 2 allow asymmetric squashing.
    """
    n = len(params)
    a1 = params[0]
    b1 = params[1]
    a2 = params[2]
    b2 = params[3]
    a3 = params[4] if n > 4 else 0.0
    b3 = params[5] if n > 5 else 0.0
    
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Radial modulation: r(theta) = 1 + a1*cos(theta) + b1*sin(theta) + a2*cos(2*theta) + b2*sin(2*theta) + ...
    radii = (1.0 
             + a1 * np.cos(angles) + b1 * np.sin(angles)
             + a2 * np.cos(2*angles) + b2 * np.sin(2*angles)
             + a3 * np.cos(3*angles) + b3 * np.sin(3*angles))
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack((x, y))

def _score_fourier(params):
    """
    Score function for Fourier parametrization.
    """
    points = _fourier_points(params)
    return _calculate_score(points)

def _soft_score_fourier(params, beta=100.0):
    """
    Soft score function for Fourier parametrization.
    """
    points = _fourier_points(params)
    return _soft_min_score(points, beta)

def _optimize_fourier(num_modes, time_limit):
    """
    Optimize Fourier series parameters.
    num_modes: 2 (modes 1,2) or 3 (modes 1,2,3)
    """
    start_time = time.time()
    
    if num_modes == 2:
        n_params = 4
        bounds = [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)]
    elif num_modes == 3:
        n_params = 6
        bounds = [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.2, 0.2), (-0.2, 0.2)]
    else:
        n_params = 4
        bounds = [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Global search with Differential Evolution
    try:
        res = differential_evolution(
            lambda p: -_score_fourier(p),
            bounds,
            seed=42,
            maxiter=80,
            popsize=20,
            tol=1e-8,
            polish=False
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Phase 2: Local polish with Soft-Min L-BFGS-B
    if time.time() - start_time < time_limit * 0.6 and best_params is not None:
        starts = [best_params]
        
        # Add some heuristic starts
        for a1 in [-0.2, 0, 0.2]:
            for a2 in [-0.2, 0, 0.2]:
                starts.append([a1, 0, a2, 0] + ([0, 0] if num_modes == 3 else []))
        
        for start in starts:
            if time.time() - start_time > time_limit * 0.85:
                break
            try:
                for beta in [50.0, 150.0, 300.0]:
                    if time.time() - start_time > time_limit * 0.85:
                        break
                    res_lbfgs = minimize(
                        lambda p, b=beta: -_soft_score_fourier(p, b),
                        start,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    
                    exact_score = _score_fourier(res_lbfgs.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_params = res_lbfgs.x
                    start = res_lbfgs.x
            except Exception:
                pass
    
    # Phase 3: High-precision Nelder-Mead on exact objective
    if time.time() - start_time < time_limit * 0.98 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_score_fourier(p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 2000, 'adaptive': True}
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
            options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 3000, 'adaptive': True}
        )
        
        if time.time() < deadline:
            candidate_score = -res.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = res.x.reshape(-1, 2)
    except Exception:
        pass
    
    # Second polish pass
    if time.time() < deadline - 2:
        try:
            def neg_score2(x):
                pts = x.reshape(-1, 2)
                return -_calculate_score(pts)
            
            res2 = minimize(
                neg_score2,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-15, 'fatol': 1e-15, 'maxiter': 1500, 'adaptive': True}
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
    deadline = start_time + 145.0 # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Try Fourier series with 2 modes (4 params)
    if time.time() < deadline - 60:
        t_limit = min(50.0, deadline - time.time() - 50)
        score, params = _optimize_fourier(2, t_limit)
        if score > best_overall_score and params is not None:
            best_overall_score = score
            best_overall_points = _fourier_points(params)
    
    # Try Fourier series with 3 modes (6 params) if time permits
    if time.time() < deadline - 40 and best_overall_points is not None:
        t_limit = min(30.0, deadline - time.time() - 30)
        score, params = _optimize_fourier(3, t_limit)
        if score > best_overall_score and params is not None:
            best_overall_score = score
            best_overall_points = _fourier_points(params)
    
    # Also try the two-ring approach as fallback/alternative
    if time.time() < deadline - 30:
        # Quick two-ring optimization
        n_out, n_in = 7, 7
        g = np.gcd(n_out, n_in)
        theta_max = 2 * np.pi / g
        bounds = [(0.0, 0.999), (0.0, theta_max)]
        
        try:
            res = differential_evolution(
                lambda p: -_calculate_score(np.vstack([
                    np.column_stack((np.cos(np.linspace(0, 2*np.pi, n_out, endpoint=False)),
                                     np.sin(np.linspace(0, 2*np.pi, n_out, endpoint=False)))),
                    np.column_stack((p[0]*np.cos(np.linspace(0, 2*np.pi, n_in, endpoint=False) + p[1]),
                                     p[0]*np.sin(np.linspace(0, 2*np.pi, n_in, endpoint=False) + p[1])))
                ])),
                bounds,
                seed=123,
                maxiter=40,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            if res.success:
                r, theta = res.x
                angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
                angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
                ring_points = np.vstack([
                    np.column_stack((np.cos(angles_out), np.sin(angles_out))),
                    np.column_stack((r*np.cos(angles_in), r*np.sin(angles_in)))
                ])
                ring_score = _calculate_score(ring_points)
                if ring_score > best_overall_score:
                    best_overall_score = ring_score
                    best_overall_points = ring_points
        except Exception:
            pass
    
    # Fallback: regular 14-gon
    if best_overall_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_overall_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_overall_score = _calculate_score(best_overall_points)
    
    # Final direct polish on all 28 coordinates
    if time.time() < deadline - 5:
        polish_time = deadline - time.time() - 3
        polished_score, polished_points = _polish_points(
            best_overall_points, polish_time, start_time
        )
        if polished_score > best_overall_score:
            best_overall_score = polished_score
            best_overall_points = polished_points
    
    return best_overall_points
