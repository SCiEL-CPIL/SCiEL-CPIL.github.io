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
    
    # Generate all combinations of 3 indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
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
    Score = A_min / A_hull
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min <= 1e-10:
        return 0.0
    
    return a_min / a_hull

def _objective(x):
    """
    Objective function to minimize (negative of the score).
    Score = A_min / A_hull
    """
    points = x.reshape(13, 2)
    return -_score(points)

def _make_initial_guess(rng):
    """
    Create a structured initial guess: regular polygon vertices + interior points.
    This provides a much better starting point than random.
    """
    n = 13
    # Use 7 vertices on a circle and 6 interior points
    n_outer = 7
    n_inner = 6
    
    # Regular polygon vertices on unit circle
    angles = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Interior points: use a smaller regular polygon, rotated
    inner_angles = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner
    inner_radius = 0.4 + rng.random() * 0.2
    inner = inner_radius * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    # Add small random perturbation to break symmetry
    perturbation = rng.random((n, 2)) * 0.05
    points = np.vstack([outer, inner]) + perturbation
    
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Phase 1: Powell optimization from structured initial guesses
    # Use multiple structured starts with different rotations/scales
    num_starts = 30
    
    for i in range(num_starts):
        if time.time() > _DEADLINE:
            break
        
        # Create structured initial guess
        x0 = _make_initial_guess(rng).flatten()
        
        # Powell optimization - handles non-smooth min-of-triangles well
        try:
            res = minimize(
                _objective, 
                x0, 
                method='Powell', 
                options={
                    'maxiter': 5000, 
                    'xtol': 1e-10, 
                    'ftol': 1e-12,
                    'niter': 100
                }
            )
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res.x.reshape(n, 2)
                
        except Exception:
            continue
        
        # Check time budget - if we have good progress, do local refinement
        if best_score > 0.01 and time.time() < _DEADLINE - 120:
            # Try Nelder-Mead refinement from best point found so far
            try:
                res2 = minimize(
                    _objective, 
                    best_points.flatten(), 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 3000, 
                        'xatol': 1e-12, 
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                refined_score = -res2.fun
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = res2.x.reshape(n, 2)
            except Exception:
                pass
    
    # Phase 2: If we have time, do a final refinement
    if best_points is not None and time.time() < _DEADLINE - 60:
        try:
            res3 = minimize(
                _objective, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={
                    'maxiter': 2000, 
                    'ftol': 1e-14,
                    'gtol': 1e-12
                }
            )
            final_score = -res3.fun
            if final_score > best_score:
                best_score = final_score
                best_points = res3.x.reshape(n, 2)
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to random if optimization failed completely
        best_points = rng.random((n, 2))
        
    return best_points

# EVOLVE-BLOCK-END
