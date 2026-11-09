# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # 20s safety margin

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
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
    
    # Calculate areas: 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product in 2D is a scalar: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the metric: A_min / A_hull.
    """
    a_hull = _hull_area(points)
    if a_hull <= 1e-12:
        return 0.0
    a_min = _min_triangle_area(points)
    return a_min / a_hull

def _soft_min_area(points, beta=50.0):
    """
    Compute a differentiable approximation of the minimum triangle area.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    n = len(points)
    indices = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Numerical stability: shift by min
    min_a = np.min(areas)
    if min_a < 0: min_a = 0
    # Use log-sum-exp trick
    # log(sum(exp(-beta * (a_i - min_a)))) + log(exp(-beta * min_a))
    # = log(sum(exp(-beta * (a_i - min_a)))) - beta * min_a
    # soft_min = -1/beta * (log(sum(exp(-beta * (a_i - min_a)))) - beta * min_a)
    # = min_a - 1/beta * log(sum(exp(-beta * (a_i - min_a))))
    
    # For very large beta, exp(-beta * (a_i - min_a)) can underflow for large diffs, which is good.
    # But if a_i - min_a is negative, it overflows. We ensured a_i >= min_a, so a_i - min_a >= 0.
    # So exp(-beta * (a_i - min_a)) <= 1.
    
    shifted = areas - min_a
    # Clip to avoid any tiny negative due to floating point
    shifted = np.clip(shifted, 0, None)
    
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return min_a
    
    log_sum = np.log(sum_exp)
    soft_min = min_a - (1.0 / beta) * log_sum
    
    return soft_min

def _objective_soft_min(points_flat, beta=50.0):
    """
    Objective function for minimization (we minimize negative soft-min area).
    """
    points = points_flat.reshape(13, 2)
    # Check validity
    if not np.all(np.isfinite(points)):
        return 1e6
    
    # Compute soft min area
    s = _soft_min_area(points, beta)
    
    # Penalize if points are too close (collinear risk)
    # The soft min handles this, but if it's 0, we want to push away.
    if s < 1e-6:
        # Calculate actual min area to check for degeneracy
        actual_min = _min_triangle_area(points)
        if actual_min < 1e-6:
            return 1e6 - s
    
    return -s

def _generate_regular_polygon(n=13, radius=1.0, seed=0):
    """
    Generate points on a regular n-gon.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    # Add small noise to break symmetry
    rng = np.random.default_rng(seed)
    points += rng.normal(0, 0.05, (n, 2))
    return points

def _generate_center_rings(n=13, seed=0):
    """
    Generate 1 center point, 6 in first ring, 6 in second ring.
    """
    rng = np.random.default_rng(seed)
    points = []
    # Center
    points.append([0.0, 0.0])
    # Ring 1: 6 points
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r1 = 0.5
    for a in angles1:
        points.append([r1 * np.cos(a), r1 * np.sin(a)])
    # Ring 2: 6 points (offset by 30 degrees)
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    r2 = 1.0
    for a in angles2:
        points.append([r2 * np.cos(a), r2 * np.sin(a)])
    
    points = np.array(points)
    points += rng.normal(0, 0.05, (n, 2))
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the x,y coordinates.
    """
    n = 13
    best_score = -1.0
    best_points = np.random.default_rng(0).random((n, 2))
    
    # List of initial generation strategies
    generators = [
        _generate_regular_polygon,
        _generate_center_rings
    ]
    
    current_seed = 0
    max_restarts = 200 # High cap, limited by time
    
    # We will use L-BFGS-B with a soft-min objective.
    # To handle the ratio, we normalize the points to have a fixed hull area (e.g., 1.0) 
    # or simply maximize the min area, since the ratio is scale invariant.
    # Maximizing min area is equivalent to maximizing the ratio if we fix the scale.
    # We will fix the scale by normalizing the points such that the convex hull area is 1.0
    # after each optimization step, or we can just maximize the min area and let the hull be whatever.
    # Actually, maximizing min area directly is not the same as maximizing the ratio if the hull area can shrink.
    # But if we shrink the hull, the min area shrinks too.
    # Let's stick to maximizing the ratio directly using a surrogate.
    # Surrogate for ratio: -soft_min_area / hull_area.
    # This is differentiable if we approximate hull_area gradient (which is hard).
    # Alternative: Fix the hull area to 1. Then maximize soft_min_area.
    
    # Let's try fixing the hull area to 1.
    # We can do this by scaling the points after each step.
    
    while time.time() < _DEADLINE and current_seed < max_restarts:
        # Check deadline
        if time.time() > _DEADLINE - 5:
            break
            
        # Cycle through generators
        gen_idx = current_seed % len(generators)
        gen_func = generators[gen_idx]
        
        # Use a unique seed for each restart to ensure diversity
        restart_seed = current_seed * 12345 + 67890
        
        try:
            initial_points = gen_func(n=n, seed=restart_seed)
            
            # Normalize initial points to unit hull area
            hull_area_init = _hull_area(initial_points)
            if hull_area_init > 0:
                initial_points = initial_points * np.sqrt(1.0 / hull_area_init)
            
            initial_flat = initial_points.flatten()
            
            # Run L-BFGS-B optimization
            # We maximize soft_min_area, so we minimize -soft_min_area
            res = minimize(
                _objective_soft_min, 
                initial_flat, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
        except Exception:
            current_seed += 1
            continue
        
        res_points = res.x.reshape(n, 2)
        
        # Validate result
        if not np.all(np.isfinite(res_points)):
            current_seed += 1
            continue
            
        # The optimization maximized the min area (approx). 
        # We should check the actual score (ratio).
        # Since we didn't explicitly constrain hull area in the optimizer, 
        # the optimizer might have shrunk the hull to increase min area? 
        # No, min area scales with square of length. Hull area also scales with square.
        # If we shrink points, both decrease. The ratio stays the same.
        # So the optimizer is indifferent to scale.
        # We should normalize the result to unit hull area to ensure consistency.
        
        hull_area_res = _hull_area(res_points)
        if hull_area_res > 0:
            res_points = res_points * np.sqrt(1.0 / hull_area_res)
        
        res_score = _score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points.copy()
            
        current_seed += 1
        
    return best_points

# EVOLVE-BLOCK-END
