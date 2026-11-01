"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_score(params):
    """Compute score for given (r_inner, angle_offset) parameters."""
    r_in, dtheta = params
    # Outer heptagon vertices
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    
    # Inner heptagon vertices
    angles_inner = angles_outer + dtheta
    inner = r_in * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    
    # Combine points
    pts = np.vstack([outer, inner])
    
    # Convex hull area (should be outer heptagon area)
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume = area
    except:
        return 0.0
    
    if hull_area <= 0:
        return 0.0
    
    # Compute all triangle areas
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Triangle area = 0.5 * |cross(pj-pi, pk-pi)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial guess: r_inner around 0.5, offset around pi/7
    best_params = [0.5, np.pi/7]
    best_score = _compute_score(best_params)
    
    # Try multiple starting points
    start_points = [
        [0.3, 0.0],
        [0.4, np.pi/14],
        [0.5, np.pi/7],
        [0.5, np.pi/6],
        [0.6, np.pi/7],
        [0.5, 0.0],
        [0.4, np.pi/7],
        [0.6, 0.0],
    ]
    
    for sp in start_points:
        if time.time() > deadline - 5.0:
            break
        
        res = minimize(
            lambda x: -_compute_score(x),
            x0=sp,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 2.0:
        res = minimize(
            lambda x: -_compute_score(x),
            x0=best_params,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x
    
    # Construct final points
    r_in, dtheta = best_params
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    angles_inner = angles_outer + dtheta
    inner = r_in * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    pts = np.vstack([outer, inner])
    
    return pts
# EVOLVE-BLOCK-END
