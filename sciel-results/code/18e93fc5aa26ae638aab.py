import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 14

# Precompute triple indices for N=14 to speed up area calculation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
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
    Soft-min surrogate for the minimum triangle area.
    Uses log-sum-exp for numerical stability.
    """
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    areas = 0.5 * np.abs(cross)

    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_area
    soft_min = max_area - log_sum / beta

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    return soft_min / hull_area

def _generate_ring_points(n_out, n_in, r, theta):
    """
    Generate points for a two-ring configuration.
    Outer ring: n_out points on unit circle.
    Inner ring: n_in points on circle of radius r, rotated by theta.
    """
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)

    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)

    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _score_ring_params(n_out, n_in, params):
    """
    Objective function for optimization.
    params: [r, theta]
    """
    r, theta = params
    if r < 0.01 or r > 0.99:
        return 0.0
    points = _generate_ring_points(n_out, n_in, r, theta)
    return _calculate_score(points)

def _soft_score_ring_params(n_out, n_in, params, beta=100.0):
    """
    Soft-min surrogate objective for gradient-based optimization.
    params: [r, theta]
    """
    r, theta = params
    if r < 0.01 or r > 0.99:
        return 0.0
    points = _generate_ring_points(n_out, n_in, r, theta)
    return _soft_min_score(points, beta)

def _optimize_ring(n_out, n_in, time_limit, deadline):
    """
    Optimize a specific ring configuration.
    """
    start_time = time.time()
    best_score = 0.0
    best_params = None

    # Symmetry: theta is periodic with 2*pi/gcd(n_out, n_in)
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g

    bounds = [(0.01, 0.99), (0.0, theta_max)]

    # Phase 1: Coarse Grid Search to find promising regions
    # This is faster and more robust than DE for 2D parameter space
    r_grid = np.linspace(0.2, 0.9, 15)
    theta_grid = np.linspace(0, theta_max, 15)
    
    for r in r_grid:
        if time.time() > deadline - 10:
            break
        for th in theta_grid:
            if time.time() > deadline - 10:
                break
            score = _score_ring_params(n_out, n_in, [r, th])
            if score > best_score:
                best_score = score
                best_params = [r, th]

    # Phase 2: Differential Evolution for global refinement
    if time.time() < deadline - 20:
        try:
            # Initialize DE around the best grid point if found
            initial_bounds = bounds
            if best_params is not None:
                # Tighten bounds around best grid point for DE
                r_c, th_c = best_params
                initial_bounds = [
                    (max(0.01, r_c - 0.2), min(0.99, r_c + 0.2)),
                    (max(0.0, th_c - theta_max/4), min(theta_max, th_c + theta_max/4))
                ]
            
            res = differential_evolution(
                lambda p: -_score_ring_params(n_out, n_in, p),
                initial_bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-5,
                polish=True
            )
            if res.success and -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass

    # Phase 3: Gradient-based polish with Soft-Min Surrogate
    if time.time() < deadline - 5 and best_params is not None:
        try:
            current_params = best_params.copy()
            # Increase beta to sharpen the minimum
            betas = [50.0, 100.0, 200.0, 500.0, 1000.0]
            
            for beta in betas:
                if time.time() > deadline - 5:
                    break
                res_soft = minimize(
                    lambda p: -_soft_score_ring_params(n_out, n_in, p, beta),
                    current_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 500}
                )
                if -res_soft.fun > best_score:
                    best_score = -res_soft.fun
                    best_params = res_soft.x
                current_params = res_soft.x
        except Exception:
            pass

    # Phase 4: Final exact evaluation and local Nelder-Mead polish
    if time.time() < deadline - 2 and best_params is not None:
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Candidate configurations (n_out, n_in)
    # 7-7 is the most symmetric and likely optimal for N=14
    # 8-6 and 6-8 are next candidates
    candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9)
    ]

    best_overall_score = 0.0
    best_overall_points = None

    for n_out, n_in in candidates:
        if time.time() > deadline - 5:
            break

        # Allocate time based on likelihood
        if (n_out, n_in) == (7, 7):
            t_limit = 60.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 30.0
        else:
            t_limit = 10.0

        score, params = _optimize_ring(n_out, n_in, t_limit, deadline)

        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r, theta = params
                best_overall_points = _generate_ring_points(n_out, n_in, r, theta)

    # Fallback if no valid points found
    if best_overall_points is None:
        n_out, n_in = 7, 7
        r, theta = 0.5, 0.0
        best_overall_points = _generate_ring_points(n_out, n_in, r, theta)

    return best_overall_points
