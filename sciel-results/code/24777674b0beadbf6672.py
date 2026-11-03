import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import random

N = 13

# Precompute triangle indices
TRI_INDICES = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            TRI_INDICES.append((i, j, k))
TRI_INDICES = np.array(TRI_INDICES)

def _get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        # Degenerate case, e.g., all points collinear
        return 0.0

    # Calculate triangle areas
    # points[:, i] etc.
    p_i = points[TRI_INDICES[:, 0]]
    p_j = points[TRI_INDICES[:, 1]]
    p_k = points[TRI_INDICES[:, 2]]
    
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    # Cross product in 2D is x1*y2 - y1*x2
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _negative_score(points_flat):
    """
    Objective function for minimizer (minimizes negative score).
    points_flat: 1D array of length 2*N
    """
    points = points_flat.reshape(N, 2)
    score = _get_score(points)
    return -score

def _fermat_spiral(n, rotation=0.0, jitter=0.0):
    """
    Generate points using Fermat spiral (Vogel's model).
    """
    i = np.arange(n)
    r = np.sqrt(i / n)
    theta = i * 2.399963229728653 + rotation
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    if jitter > 0:
        x += np.random.normal(0, jitter, n)
        y += np.random.normal(0, jitter, n)
        
    points = np.column_stack((x, y))
    # Center the points
    points -= np.mean(points, axis=0)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Initial guess: Fermat Spiral
    # We will try several restarts with different rotations and jitters
    n_restarts = 30  # Max number of restarts we *want* to try
    
    for r_idx in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Create initial configuration
        # Vary rotation and add some jitter to escape symmetry
        rotation = 2 * np.pi * r_idx / n_restarts
        jitter = 0.01 if r_idx > 0 else 0.0
        
        init_points = _fermat_spiral(N, rotation=rotation, jitter=jitter)
        
        # Flatten for optimizer
        init_flat = init_points.flatten()
        
        # Run Nelder-Mead optimization
        # maxiter is set to allow for some progress but not hang
        # We can adjust maxiter based on remaining time
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
            
        # Estimate iterations: Nelder-Mead is slow-ish. 
        # Let's allow up to 200 iterations per restart, or until time is low.
        max_iter = 200
        if remaining_time < 20:
            max_iter = 50
            
        try:
            res = minimize(
                _negative_score, 
                init_flat, 
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            curr_score = -res.fun
            curr_points = res.x.reshape(N, 2)
            
            if curr_score > best_score:
                best_score = curr_score
                best_points = curr_points
                
        except Exception:
            continue
            
        # If we have a very good score, we might do a final polish
        # But for now, just continue with next restart
        
    if best_points is None:
        # Fallback to a basic Fermat spiral if optimization failed completely
        best_points = _fermat_spiral(N)
        
    return best_points
