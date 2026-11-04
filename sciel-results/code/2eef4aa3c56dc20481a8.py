import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 14
SOFT_BETA = 50.0  # Temperature for soft-min; higher = sharper min approximation

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

def _soft_min_score(points, beta=SOFT_BETA):
    """
    Smooth surrogate: -soft_min of triangle areas / hull area.
    Returns a smooth value that approximates -min_area/hull_area.
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
    
    # Soft-min via log-sum-exp: soft_min(a) ≈ (1/beta) * log(sum(exp(-beta * a)))
    # This is smooth and approximates min(a) as beta -> inf
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_min_val / hull_area


def _get_fourier_points(params):
    """
    Generate 14 points using a Fourier series for the radius.
    params: [a1, b1, a2]
    r_k = 1 + a1 * cos(theta_k) + b1 * sin(theta_k) + a2 * cos(2*theta_k)
    We fix b2=0 for simplicity (symmetry).
    We assume the points are ordered by angle theta_k = 2*pi*k/14.
    """
    a1, b1, a2 = params
    
    # Ensure radii are positive
    # If r becomes negative, the point flips to the opposite side, which might be valid
    # but usually we want a star-shaped polygon around the origin.
    # Let's clip r to be > 0.1 to avoid degenerate cases near origin.
    
    k = np.arange(N)
    theta = 2 * np.pi * k / N
    
    r = 1.0 + a1 * np.cos(theta) + b1 * np.sin(theta) + a2 * np.cos(2 * theta)
    
    # Clip to ensure we don't hit the origin or go negative (which flips direction)
    # If we allow negative r, the point is at (r*cos, r*sin) which is same as (|r|*cos(theta+pi), |r|*sin(theta+pi))
    # This breaks the "regular polygon" ordering assumption if we want to maintain convexity easily.
    # Let's just clip to small positive value to maintain simple star shape.
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack((x, y))

def _score_fourier(params):
    """
    Score for Fourier parameterization.
    """
    points = _get_fourier_points(params)
    return _calculate_score(points)

def _soft_min_fourier(params, beta=SOFT_BETA):
    """
    Soft-min score for Fourier parameterization.
    """
    points = _get_fourier_points(params)
    return _soft_min_score(points, beta=beta)

def _optimize_fourier(time_limit, start_time, deadline):
    """
    Optimize the Fourier parameters.
    """
    # Bounds for a1, b1, a2
    # We want to allow significant deformation but not collapse.
    # r_min = 1 - |a1| - |b1| - |a2| > 0.1
    # So |a1| + |b1| + |a2| < 0.9
    bounds = [(-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4)]
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Phase 1: Differential Evolution on Fourier params
    # 3 parameters, so DE should be fast.
    popsize = 20
    maxiter = 50
    
    try:
        res = differential_evolution(
            lambda p: _soft_min_fourier(p, beta=SOFT_BETA), # Use soft min for smooth landscape
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=True
        )
        
        if res.success:
            best_params = res.x
            best_points = _get_fourier_points(best_params)
            best_score = _calculate_score(best_points)
    except Exception as e:
        pass
    
    # Phase 2: Local polish with multiple starts using exact score
    if time.time() < deadline - 10.0 and best_params is not None:
        # Start from the DE result and some perturbations
        starts = [best_params.copy()]
        for dim in range(3):
            for delta in [-0.1, 0.1]:
                p = best_params.copy()
                p[dim] += delta
                starts.append(p)
        
        for start in starts:
            if time.time() > deadline - 10.0:
                break
            try:
                res_nm = minimize(
                    lambda p: -_score_fourier(p),
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x
                    best_points = _get_fourier_points(best_params)
            except Exception:
                pass
                
    # Phase 3: Coordinate-level polish to escape Fourier parameterization
    if best_points is not None and time.time() < deadline - 5.0:
        # Use soft-min for smooth gradient, then verify with exact min
        def neg_soft_score(x):
            pts = x.reshape(-1, 2)
            return _soft_min_score(pts, beta=SOFT_BETA)
        
        try:
            res = minimize(
                neg_soft_score,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            polished = res.x.reshape(-1, 2)
            polished_score = _calculate_score(polished)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
                
        except Exception:
            pass
            
        # Second polish with higher beta
        if time.time() < deadline - 3.0:
            def neg_soft_score_high(x):
                pts = x.reshape(-1, 2)
                return _soft_min_score(pts, beta=SOFT_BETA * 5)
            
            try:
                res2 = minimize(
                    neg_soft_score_high,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                polished2 = res2.x.reshape(-1, 2)
                polished2_score = _calculate_score(polished2)
                
                if polished2_score > best_score:
                    best_score = polished2_score
                    best_points = polished2
                    
            except Exception:
                pass
    
    return best_score, best_points

def _get_points_2p(n_out, n_in, r, theta):
    """
    Generate points for the 2-parameter ring configuration.
    """
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _optimize_2p(n_out, n_in, time_limit, start_time, deadline):
    """
    Optimize the 2-parameter ring configuration (r, theta).
    """
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Phase 1: Differential Evolution with soft-min surrogate for smoothness
    popsize = 10
    maxiter = 25
    
    if time_limit > 30:
        popsize = 15
        maxiter = 40
    elif time_limit > 15:
        popsize = 12
        maxiter = 35

    try:
        res = differential_evolution(
            lambda p: _soft_min_ring_score(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=True
        )
        
        if res.success:
            best_params = res.x
            r, theta = best_params
            best_points = _get_points_2p(n_out, n_in, r, theta)
            best_score = _calculate_score(best_points)
    except Exception as e:
        pass
    
    # Phase 2: Local polish with multiple starts using exact score
    if time.time() < deadline - 8.0:
        starts = []
        for r_start in [0.3, 0.5, 0.7, 0.9, 0.95]:
            for theta_start in [0.0, theta_max/2, theta_max/3, 2*theta_max/3]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() > deadline - 8.0:
                break
            try:
                res_nm = minimize(
                    lambda p: -_score_2p_params(p, n_out, n_in),
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 150}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x
                    r, theta = best_params
                    best_points = _get_points_2p(n_out, n_in, r, theta)
            except Exception:
                pass
    
    # Phase 3: Coordinate-level polish to escape ring parameterization
    if best_points is not None and time.time() < deadline - 5.0:
        # Use soft-min for smooth gradient, then verify with exact min
        def neg_soft_score(x):
            pts = x.reshape(-1, 2)
            return _soft_min_score(pts, beta=SOFT_BETA)
        
        try:
            res = minimize(
                neg_soft_score,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            polished = res.x.reshape(-1, 2)
            polished_score = _calculate_score(polished)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
                
        except Exception:
            pass
            
        # Second polish with higher beta
        if time.time() < deadline - 3.0:
            def neg_soft_score_high(x):
                pts = x.reshape(-1, 2)
                return _soft_min_score(pts, beta=SOFT_BETA * 5)
            
            try:
                res2 = minimize(
                    neg_soft_score_high,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                polished2 = res2.x.reshape(-1, 2)
                polished2_score = _calculate_score(polished2)
                
                if polished2_score > best_score:
                    best_score = polished2_score
                    best_points = polished2
                    
            except Exception:
                pass
        
    if best_points is None:
        return 0.0, None
        
    # Return the actual polished points as a flat array for the caller to reshape
    return best_score, best_points.flatten()

def _soft_min_ring_score(n_out, n_in, params):
    """
    Smooth surrogate objective for ring parameters using soft-min.
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
    
    return _soft_min_score(points)

def _score_2p_params(params, n_out, n_in):
    """
    Objective for 2-parameter optimization: r and theta.
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
    
    return _calculate_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # 1. Try Fourier Series Optimization (New Strategy)
    # This allows for more flexible shapes than strict rings.
    if time.time() < deadline - 30.0:
        fourier_time = min(40.0, deadline - 30.0 - time.time())
        score_f, points_f = _optimize_fourier(fourier_time, start_time, deadline)
        if points_f is not None and score_f > best_overall_score:
            best_overall_score = score_f
            best_overall_points = points_f
            
    # 2. Try Ring Optimizations (Fallback / Comparison)
    # We prioritize the most symmetric splits.
    candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9)
    ]
    
    for n_out, n_in in candidates:
        if time.time() > deadline - 10:
            break
            
        # Time budget for this candidate
        if (n_out, n_in) == (7, 7):
            t_limit = 20.0
        else:
            t_limit = 10.0
            
        if time.time() > deadline - 10:
            break
            
        score, params = _optimize_2p(n_out, n_in, t_limit, start_time, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                best_overall_points = params.reshape(-1, 2)
                actual_score = _calculate_score(best_overall_points)
                best_overall_score = actual_score
                
    # Fallback if no valid points found
    if best_overall_points is None:
        # Return a regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        best_overall_points = np.column_stack((x, y))
        
    return best_overall_points
