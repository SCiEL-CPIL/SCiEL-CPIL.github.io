"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _decode_central_symmetric(params):
    """
    Decode parameters into 14 points forming a centrally symmetric convex polygon.
    params: array of shape (14,)
      - first 7: radial distances r_0..r_6 (must be > 0)
      - last 7: angle offsets d_0..d_6 (added to base angles)
    Base angles: 7 evenly spaced angles over [0, pi)
    Points: for each i in 0..6, point i = r_i * (cos(theta_i), sin(theta_i))
            point i+7 = -point i
    """
    r = np.exp(params[:7])  # Ensure positive radii via exp
    d = params[7:]
    
    base_angles = np.linspace(0, np.pi, 7, endpoint=False)
    angles = base_angles + d
    
    # Create 7 points on one half
    half_points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    # Full set: 7 points + their negatives
    points = np.vstack([half_points, -half_points])
    
    # Sort by angle to ensure convex polygon order
    point_angles = np.arctan2(points[:, 1], points[:, 0])
    order = np.argsort(point_angles)
    points = points[order]
    
    return points


def _objective_cs(params):
    """Objective for centrally symmetric optimization: minimize negative score."""
    points = _decode_central_symmetric(params)
    s = _score(points)
    return -s


def _objective_cs_soft(params):
    """Soft objective: penalize non-convexity."""
    points = _decode_central_symmetric(params)
    s = _score(points)
    
    # Check convexity: all cross products of consecutive edges should have same sign
    n = len(points)
    cross_products = []
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        cross_products.append(cross)
    
    cross_products = np.array(cross_products)
    # If any cross product is negative (assuming CCW orientation), penalty
    penalty = np.sum(np.minimum(0, cross_products)) ** 2
    
    return -s + 10 * penalty


def _regular_polygon_init():
    """Generate initial parameters for a regular 14-gon."""
    r = np.ones(7)
    d = np.zeros(7)
    return np.concatenate([np.log(r), d])


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with regular polygon
    best_params = _regular_polygon_init()
    best_points = _decode_central_symmetric(best_params)
    best_score = _score(best_points)
    
    # Define bounds for differential evolution
    # r in [0.5, 2.0] -> log(r) in [-0.69, 0.69]
    # d in [-pi/7, pi/7] (to keep angles ordered)
    bounds = [(-0.69, 0.69)] * 7 + [(-0.45, 0.45)] * 7
    
    # Try differential evolution
    de_kwargs_list = [
        {'maxiter': 100, 'popsize': 15, 'tol': 1e-5, 'seed': 42, 'init': 'latinhypercube'},
        {'maxiter': 80, 'popsize': 12, 'tol': 1e-6, 'seed': 123, 'init': 'latinhypercube'},
    ]
    
    for kwargs in de_kwargs_list:
        if time.time() > deadline - 40:
            break
        
        try:
            result = differential_evolution(
                _objective_cs_soft, bounds,
                maxiter=kwargs['maxiter'],
                popsize=kwargs['popsize'],
                tol=kwargs['tol'],
                seed=kwargs['seed'],
                polish=False,
                init=kwargs['init']
            )
            
            candidate_params = result.x
            candidate_points = _decode_central_symmetric(candidate_params)
            candidate_score = _score(candidate_points)
            
            if candidate_score > best_score:
                best_params = candidate_params.copy()
                best_points = candidate_points.copy()
                best_score = candidate_score
                
        except Exception:
            continue
    
    # Refine with L-BFGS-B on the hard objective
    if time.time() < deadline - 20:
        try:
            result = minimize(
                _objective_cs, best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            candidate_params = result.x
            candidate_points = _decode_central_symmetric(candidate_params)
            candidate_score = _score(candidate_points)
            
            if candidate_score > best_score:
                best_params = candidate_params.copy()
                best_points = candidate_points.copy()
                best_score = candidate_score
                
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
