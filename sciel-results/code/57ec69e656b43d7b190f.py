"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
DEADLINE = 140.0  # safety margin before 150s kill

# Precompute all C(11,3) = 165 triplets
_TRIPLES = np.array(list(itertools.combinations(range(N), 3)))
_T_I = _TRIPLES[:, 0]
_T_J = _TRIPLES[:, 1]
_T_K = _TRIPLES[:, 2]


def _min_area(points):
    """Minimum triangle area among all C(N,3) triples."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                         (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return float(areas.min())


def _params_to_points(params):
    """
    Maps 11 parameters to 11 points in the triangle, enforcing reflection symmetry.
    params[0]: y-coordinate of the central point (x=0.5)
    params[1:]: x, u for 5 pairs. x in [0, 0.5], u in [0, 1]. y = u * sqrt(3) * x.
    
    This parameterization guarantees all points are inside the triangle:
    - Central point: (0.5, y0) with y0 in [0, H]
    - For each pair i: left point (x_i, u_i * sqrt(3) * x_i), right point (1-x_i, u_i * sqrt(3) * x_i)
      Since x_i in [0, 0.5] and u_i in [0, 1], y = u_i * sqrt(3) * x_i <= sqrt(3) * x_i (left edge)
      And by symmetry, y <= sqrt(3) * (1-x_i) (right edge) is also satisfied.
      Also y >= 0 and y <= H automatically.
    """
    points = np.zeros((N, 2))
    
    # Central point (index 0)
    points[0, 0] = 0.5
    points[0, 1] = params[0]
    
    # 5 pairs (indices 1-10)
    for i in range(5):
        x = params[1 + 2 * i]
        u = params[2 + 2 * i]
        y = u * np.sqrt(3) * x
        
        # Left point
        points[1 + 2 * i, 0] = x
        points[1 + 2 * i, 1] = y
        
        # Right point (reflection)
        points[2 + 2 * i, 0] = 1.0 - x
        points[2 + 2 * i, 1] = y
        
    return points


def _objective(params):
    """Objective function to minimize (negative of min triangle area)."""
    points = _params_to_points(params)
    return -_min_area(points)


def _soft_min_area(params, beta=30.0):
    """
    Differentiable soft-minimum of triangle areas using log-sum-exp.
    Returns a smooth approximation of -min(areas).
    """
    points = _params_to_points(params)
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    max_area = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_area))))
    soft_min_val = -(log_sum + beta * max_area) / beta
    
    return soft_min_val


def _objective_soft(params, beta=30.0):
    """Differentiable objective using soft-min for gradient-based optimization."""
    return _soft_min_area(params, beta)


def _local_refine(params, time_budget, rng):
    """
    Local refinement using multiple strategies:
    1. Gradient-based with soft-min objective
    2. Nelder-Mead with exact objective
    3. Iterative noise-based search with decreasing step size
    """
    start_time = time.time()
    best_params = params.copy()
    best_score = -_objective(params)
    
    # Bounds
    lb = np.zeros(11)
    ub = np.zeros(11)
    ub[0] = H
    ub[1::2] = 0.5  # x coordinates
    ub[2::2] = 1.0  # u coordinates
    
    bounds = list(zip(lb, ub))
    
    # Strategy 1: Gradient-based with soft-min
    if time.time() - start_time < time_budget * 0.4:
        try:
            res = minimize(_objective_soft, best_params, method='L-BFGS-B',
                           bounds=bounds, args=(30.0,),
                           options={'maxiter': 200, 'maxfev': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < 1e9:
                score = -_objective(res.x)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Nelder-Mead with exact objective
    if time.time() - start_time < time_budget * 0.7:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'maxfev': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            # Check bounds
            if np.all(res.x >= lb - 1e-9) and np.all(res.x <= ub + 1e-9):
                score = -_objective(res.x)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Iterative noise-based search
    if time.time() - start_time < time_budget:
        step_size = 0.05
        while time.time() - start_time < time_budget:
            noise = rng.normal(0, step_size, size=11)
            new_params = best_params + noise
            new_params = np.clip(new_params, lb, ub)
            
            score = -_objective(new_params)
            if score > best_score:
                best_score = score
                best_params = new_params.copy()
                step_size = max(0.001, step_size * 0.99)
            else:
                step_size = max(0.001, step_size * 0.95)
    
    return best_params, best_score


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Bounds for DE
    lb = np.zeros(11)
    ub = np.zeros(11)
    ub[0] = H
    ub[1::2] = 0.5  # x coordinates
    ub[2::2] = 1.0  # u coordinates
    
    bounds = list(zip(lb, ub))
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Global Search with Differential Evolution
    time_left = DEADLINE - time.time() - 20
    if time_left > 30:
        try:
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=150,
                popsize=20,
                seed=42,
                tol=1e-6,
                polish=False,
                update_interval=5,
                callback=lambda x, convergence: (time.time() - start_time > DEADLINE - 25)
            )
            best_params = result.x.copy()
            best_score = -result.fun
        except Exception:
            # Fallback to random
            best_params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
            best_score = -_objective(best_params)
    
    if best_params is None:
        best_params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        best_score = -_objective(best_params)
    
    # Phase 2: Local refinement on the best solution
    time_left = DEADLINE - time.time() - 10
    if time_left > 15:
        params, score = _local_refine(best_params, time_left, rng)
        if score > best_score:
            best_score = score
            best_params = params
    
    # Phase 3: A few more random restarts with local refinement if time permits
    for _ in range(5):
        time_left = DEADLINE - time.time() - 10
        if time_left < 15:
            break
        try:
            random_params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
            params, score = _local_refine(random_params, min(10.0, time_left - 5), rng)
            if score > best_score:
                best_score = score
                best_params = params
        except Exception:
            pass
    
    # Final polish
    time_left = DEADLINE - time.time() - 5
    if time_left > 5:
        try:
            params, score = _local_refine(best_params, time_left, rng)
            if score > best_score:
                best_score = score
                best_params = params
        except Exception:
            pass
    
    points = _params_to_points(best_params)
    return points
