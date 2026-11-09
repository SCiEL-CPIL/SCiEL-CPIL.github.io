# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 575.0  # Safety margin before 630s limit

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _soft_min_triangle_area(points, temperature=0.01):
    """
    Soft-min surrogate for the minimum triangle area using log-sum-exp.
    This provides a smooth, differentiable approximation to the min operation.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp for soft min: -t * log(sum(exp(-areas/t)))
    # As t -> 0, this approaches the true minimum
    min_area = np.min(areas)
    # Shift for numerical stability
    shifted = -areas / temperature
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min = -temperature * log_sum_exp
    
    return soft_min

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
    Score = A_min / A_hull
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min <= 1e-10:
        return 0.0
    
    return a_min / a_hull

def _objective_smooth(x, temperature=0.01):
    """
    Smooth objective using soft-min surrogate.
    Minimizes negative of (soft_min_area / hull_area).
    """
    points = x.reshape(13, 2)
    a_min_soft = _soft_min_triangle_area(points, temperature)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min_soft <= 1e-10:
        return 0.0
    
    return -(a_min_soft / a_hull)

def _objective_exact(x):
    """
    Exact objective (non-smooth) for final refinement.
    """
    points = x.reshape(13, 2)
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min <= 1e-10:
        return 0.0
    
    return -(a_min / a_hull)

def _make_initial_guess(ring_config, angle_offsets=None, rng=None):
    """
    Create an initial guess based on concentric ring topology.
    
    ring_config: list of (radius, num_points) tuples
    angle_offsets: optional list of angle offsets for each ring
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    points = []
    for ring_idx, (radius, num_points) in enumerate(ring_config):
        if angle_offsets is not None and ring_idx < len(angle_offsets):
            offset = angle_offsets[ring_idx]
        else:
            offset = 0.0
        
        for k in range(num_points):
            angle = 2 * np.pi * k / num_points + offset
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    return np.array(points)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    
    Uses two-phase optimization:
    1. Smooth surrogate with Powell's method from structured initial guesses
    2. Exact refinement with Nelder-Mead
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Define several structured ring topologies for 13 points
    # Various configurations: center + rings
    ring_configs = [
        # (radius, num_points) tuples
        [(0, 1), (0.5, 6), (1.0, 6)],           # center + 2 rings
        [(0, 1), (0.4, 4), (0.8, 4), (1.2, 4)], # center + 3 rings
        [(0, 1), (0.3, 3), (0.6, 3), (0.9, 3), (1.2, 3)],  # center + 4 rings
        [(0.3, 5), (0.8, 8)],                     # 2 rings only
        [(0.2, 3), (0.5, 5), (1.0, 5)],           # 3 rings
        [(0, 1), (0.7, 12)],                      # center + 1 ring
        [(0.4, 6), (0.9, 7)],                     # 2 rings
        [(0, 1), (0.3, 5), (0.7, 7)],             # center + 2 rings
    ]
    
    # Phase 1: Smooth optimization from structured starts
    num_smooth_restarts = 30
    
    for i in range(num_smooth_restarts):
        if time.time() > _DEADLINE:
            break
        
        # Choose a ring configuration
        config_idx = i % len(ring_configs)
        ring_config = ring_configs[config_idx]
        
        # Add some random perturbation to angles
        angle_offsets = [rng.random() * 2 * np.pi for _ in ring_config]
        
        # Create initial guess
        x0_pts = _make_initial_guess(ring_config, angle_offsets, rng)
        
        # Add small random noise
        x0_pts = x0_pts + rng.normal(0, 0.05, x0_pts.shape)
        
        x0 = x0_pts.flatten()
        
        # Optimize with smooth objective using Powell's method
        try:
            res = minimize(
                _objective_smooth,
                x0,
                method='Powell',
                args=(0.01,),
                options={
                    'maxiter': 500,
                    'xtol': 1e-6,
                    'ftol': 1e-8,
                }
            )
            
            # Evaluate exact score at this point
            current_pts = res.x.reshape(n, 2)
            current_score = _score(current_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
                
        except Exception:
            continue
    
    # Phase 2: Exact refinement from best point found
    if best_points is not None and time.time() < _DEADLINE:
        x0 = best_points.flatten()
        
        try:
            res = minimize(
                _objective_exact,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            current_pts = res.x.reshape(n, 2)
            current_score = _score(current_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
                
        except Exception:
            pass
    
    # Additional random restarts if time permits
    remaining_time = _DEADLINE - time.time()
    if remaining_time > 60:
        num_extra = min(50, int(remaining_time / 10))
        
        for i in range(num_extra):
            if time.time() > _DEADLINE:
                break
            
            x0 = rng.random((n, 2)).flatten()
            
            try:
                res = minimize(
                    _objective_smooth,
                    x0,
                    method='Powell',
                    args=(0.01,),
                    options={
                        'maxiter': 300,
                        'xtol': 1e-5,
                        'ftol': 1e-7,
                    }
                )
                
                current_pts = res.x.reshape(n, 2)
                current_score = _score(current_pts)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
                    
            except Exception:
                continue
    
    if best_points is None:
        best_points = rng.random((n, 2))
    
    return best_points

# EVOLVE-BLOCK-END
