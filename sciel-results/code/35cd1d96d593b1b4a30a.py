"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation for all triples
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    # We iterate over pairs (i, j) and vectorize over k
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _negative_score(params):
    """
    Objective function for optimization.
    params: [r_inner, theta_offset]
    """
    r_inner, theta_offset = params
    if r_inner < 0.01 or r_inner > 0.99:
        return 1e9 # Penalize invalid radii
    
    # Construct points
    # Outer ring: 7 points on unit circle
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    # Inner ring: 7 points on circle of radius r_inner, rotated by theta_offset
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta_offset
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    
    # The score is scale invariant, so we don't need to normalize further
    # but we ensure the hull is computed correctly.
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin

    # Phase 1: Coarse Grid Search
    # We expect the optimal r_inner to be around 0.5 and rotation to be pi/7 (interleaved)
    # Let's search r in [0.2, 0.8] and theta in [0, 2*pi]
    
    best_score = -np.inf
    best_params = None
    
    # Grid size: 50x50 = 2500 evaluations. Each eval is fast (<1ms).
    # 2500 * 1ms = 2.5s. Well within budget.
    r_vals = np.linspace(0.2, 0.8, 50)
    theta_vals = np.linspace(0, 2*np.pi, 50, endpoint=False)
    
    for r in r_vals:
        for theta in theta_vals:
            params = np.array([r, theta])
            neg_s = _negative_score(params)
            s = -neg_s
            
            if s > best_score:
                best_score = s
                best_params = params.copy()
        
        if time.time() > deadline - 10:
            break

    # Phase 2: Local Refinement with Nelder-Mead
    if best_params is not None:
        try:
            res = minimize(
                _negative_score, 
                best_params, 
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500}
            )
            
            final_neg_score = res.fun
            final_score = -final_neg_score
            
            if final_score > best_score:
                best_params = res.x
                best_score = final_score
                
        except Exception:
            pass # Keep grid search result

    # Construct final points from best_params
    r_inner, theta_offset = best_params
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta_offset
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    
    # Normalize to unit area? The task says "normalized to unit area" but score is ratio.
    # The ratio is scale invariant. However, returning points in a standard scale is good practice.
    # The seed program normalized to unit RMS radius. Let's do the same for consistency.
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms > 1e-6:
        pts = pts / rms
        
    return pts
# EVOLVE-BLOCK-END
