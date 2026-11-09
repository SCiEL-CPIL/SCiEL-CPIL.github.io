# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 560.0  # 40s safety margin before 600s kill

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    return float(np.min(areas))

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the ratio A_min / A_hull.
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull < 1e-10:
        return 0.0
        
    return a_min / a_hull

def _soft_min_objective(params, beta):
    """
    Objective function using a soft-min approximation for the minimum triangle area.
    We minimize the negative of the score: -A_min / A_hull.
    """
    points = params.reshape(13, 2)
    
    # Compute all triangle areas
    n = 13
    indices = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    # Soft min approximation: -log(sum(exp(-beta * areas))) / beta
    # This is differentiable and approximates min(areas) as beta -> inf
    # To avoid overflow/underflow, we shift by the minimum
    min_area = np.min(areas)
    exp_term = np.exp(-beta * (areas - min_area))
    soft_min_val = min_area - np.log(np.sum(exp_term)) / beta
    
    a_hull = _hull_area(points)
    if a_hull < 1e-10:
        return 1e10
    
    # Maximize soft_min_val / a_hull -> Minimize -soft_min_val / a_hull
    return -soft_min_val / a_hull

def _exact_objective(params):
    """
    Exact objective function for final polish.
    """
    points = params.reshape(13, 2)
    s = _score(points)
    return -s

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius
    return np.column_stack((x, y))

def _fibonacci_spiral(n, radius=1.0):
    """
    Generate points inside a circle using a Fibonacci-like spiral for even distribution.
    """
    golden_angle = np.pi * (3 - np.sqrt(5))
    i = np.arange(n)
    r = radius * np.sqrt((i + 0.5) / n)
    theta = i * golden_angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    # 1. Generate diverse initializations
    inits = []
    inits.append(_regular_polygon(n))
    inits.append(_fibonacci_spiral(n))
    
    # Add a few random inits
    rng = np.random.default_rng(42)
    for _ in range(3):
        inits.append(rng.random((n, 2)) * 2 - 1)
        
    # Also add the regular polygon with a slight perturbation
    inits.append(_regular_polygon(n) + rng.normal(0, 0.1, (n, 2)))

    best_params = inits[0].flatten()
    best_score = _score(inits[0])
    
    # Evaluate all inits and keep the best
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_params = init.flatten()

    # 2. Optimization Strategy:
    # Phase 1: L-BFGS-B with soft-min surrogate (increasing beta)
    # Phase 2: Powell with exact objective
    
    current_params = best_params.copy()
    
    # Annealing schedule for beta
    betas = [10, 50, 200, 1000]
    
    for beta in betas:
        if time.time() > _DEADLINE - 120:
            break
            
        try:
            res = minimize(
                _soft_min_objective,
                current_params,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 20}
            )
            if res.fun < -best_score: # res.fun is -score
                current_params = res.x
                best_score = -res.fun
            # Even if not better, use the result as starting point for next beta
            # as it might be in a better basin for the sharper landscape
            if res.success:
                 current_params = res.x
        except Exception:
            pass

    # 3. Final Polish with Powell on the exact objective
    if time.time() < _DEADLINE - 60:
        try:
            res = minimize(
                _exact_objective,
                current_params,
                method='Powell',
                options={'maxiter': 500, 'maxfev': 10000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
            else:
                # Keep the best found so far, which might be from the soft-min phase
                # if Powell got stuck or didn't improve
                pass 
        except Exception:
            pass
            
    # 4. If time remains, try a few random restarts with the same pipeline
    # but shorter budgets
    while time.time() < _DEADLINE - 30:
        init = rng.random((n, 2)) * 2 - 1
        s_init = _score(init)
        if s_init < best_score * 0.8: # Heuristic: skip very bad inits
            continue
            
        p = init.flatten()
        
        # Shorter annealing
        for beta in [50, 500]:
             if time.time() > _DEADLINE - 20:
                 break
             try:
                 res = minimize(
                     _soft_min_objective,
                     p,
                     args=(beta,),
                     method='L-BFGS-B',
                     options={'maxiter': 200, 'ftol': 1e-12}
                 )
                 if res.success:
                     p = res.x
             except:
                 break
                 
        # Quick polish
        try:
            res = minimize(
                _exact_objective,
                p,
                method='Powell',
                options={'maxiter': 100, 'maxfev': 2000}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_params = res.x
        except:
            pass

    return best_params.reshape(13, 2)

# EVOLVE-BLOCK-END
