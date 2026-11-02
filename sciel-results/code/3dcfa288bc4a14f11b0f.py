"""Optimized solver for Heilbronn convex regions n=14."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14
DEADLINE = 145.0  # safety margin before 150s kill


def _min_triangle_ratio(points):
    """Compute min triangle area / convex hull area for n points."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, this is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Find minimum triangle area over all triples
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * |cross(pj - pi, pk - pi)|
                dx1 = pj[0] - pi[0]
                dy1 = pj[1] - pi[1]
                dx2 = pk[0] - pi[0]
                dy2 = pk[1] - pi[1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _neg_objective(params):
    """Negative of the ratio (we minimize this)."""
    # params: first N-1 are angles (relative), last is a scale
    # We'll construct points on a circle with these angles
    angles = params[:N - 1]
    # Add the final angle to close the polygon (should be ~2*pi)
    # Actually, let's use N angles that sum to 2*pi
    # Better: use N angles, normalize so they sum to 2*pi
    angles = np.asarray(angles, dtype=np.float64)
    if angles.sum() <= 0:
        return 1.0  # large penalty
    
    # Normalize angles to sum to 2*pi
    angles = angles / angles.sum() * 2 * np.pi
    
    # Sort angles to ensure convex ordering
    angles = np.sort(angles)
    
    # Points on unit circle
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    ratio = _min_triangle_ratio(points)
    return -ratio


def construct():
    start_time = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Initial: regular 14-gon
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    best_score = _min_triangle_ratio(points_reg)
    best_points = points_reg.copy()
    
    # Multi-start Nelder-Mead optimization
    rng = np.random.RandomState(42)
    
    # We optimize N angle parameters (they'll be normalized to sum to 2*pi)
    # Start from various perturbations of the regular configuration
    
    max_starts = 20
    start_count = 0
    
    for start_idx in range(max_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Generate initial guess
        if start_idx == 0:
            # Regular
            angles0 = angles_reg.copy()
        else:
            # Perturbed regular
            perturbation = rng.uniform(-0.3, 0.3, size=N)
            angles0 = angles_reg + perturbation
            # Ensure positive
            angles0 = np.maximum(angles0, 0.01)
        
        # Nelder-Mead
        try:
            result = minimize(
                _neg_objective,
                angles0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < 0:
                angles_opt = result.x
                angles_opt = angles_opt / angles_opt.sum() * 2 * np.pi
                angles_opt = np.sort(angles_opt)
                points_opt = np.column_stack([np.cos(angles_opt), np.sin(angles_opt)])
                
                score = _min_triangle_ratio(points_opt)
                if score > best_score:
                    best_score = score
                    best_points = points_opt.copy()
        except Exception:
            pass
        
        start_count += 1
    
    # Try a few more random starts if time permits
    for extra_start in range(10):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Random angles (positive, will be normalized)
        angles_rand = rng.uniform(0.1, 1.0, size=N)
        
        try:
            result = minimize(
                _neg_objective,
                angles_rand,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-7,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < 0:
                angles_opt = result.x
                angles_opt = angles_opt / angles_opt.sum() * 2 * np.pi
                angles_opt = np.sort(angles_opt)
                points_opt = np.column_stack([np.cos(angles_opt), np.sin(angles_opt)])
                
                score = _min_triangle_ratio(points_opt)
                if score > best_score:
                    best_score = score
                    best_points = points_opt.copy()
        except Exception:
            pass
    
    # Final check: make sure we return a valid point set
    if best_points is None:
        # Fallback: regular 14-gon
        best_points = points_reg
    
    # Verify the result
    final_score = _min_triangle_ratio(best_points)
    if final_score <= 0:
        best_points = points_reg
    
    return best_points


# EVOLVE-BLOCK-START
# The construct function above is the entry point
# EVOLVE-BLOCK-END
