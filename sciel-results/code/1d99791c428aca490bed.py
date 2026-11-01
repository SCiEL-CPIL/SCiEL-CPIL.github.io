import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3
TOTAL_TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def _compute_ratio(points):
    """
    Compute the squared ratio of max to min pairwise distance.
    points: (N, D) array
    """
    if len(points) < 2:
        return np.inf
    
    # Compute squared distances
    # Using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    
    # Mask out self-distances (diagonal)
    mask = ~np.eye(len(points), dtype=bool)
    dist_sq = dist_sq[mask]
    
    if len(dist_sq) == 0:
        return np.inf
        
    min_d = np.min(dist_sq)
    max_d = np.max(dist_sq)
    
    if min_d < 1e-12:
        return np.inf
        
    return (max_d / min_d)

def _smooth_ratio(points, tau=0.1):
    """
    Smooth approximation of the ratio using log-sum-exp.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    
    mask = ~np.eye(len(points), dtype=bool)
    dist_sq = dist_sq[mask]
    
    # Soft max
    # To avoid overflow in exp, subtract max
    max_val = np.max(dist_sq)
    soft_max = tau * np.log(np.sum(np.exp((dist_sq - max_val) / tau))) + max_val
    
    # Soft min
    min_val = np.min(dist_sq)
    soft_min = -tau * np.log(np.sum(np.exp(-(dist_sq - min_val) / tau))) + min_val
    
    if soft_min < 1e-12:
        return 1e12
        
    return soft_max / soft_min

def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME_LIMIT - SAFETY_MARGIN
    
    best_score = np.inf
    best_points = None
    
    # Initial guesses
    # 1. Random
    # 2. Tetrahedron-based (if we want to try the hint, but random is robust)
    
    n_restarts = 20 # Adjust based on time
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial points
        if i == 0:
            # Try the tetrahedron + midpoints + 4 random
            # Tetrahedron vertices
            v1 = np.array([1, 1, 1])
            v2 = np.array([1, -1, -1])
            v3 = np.array([-1, 1, -1])
            v4 = np.array([-1, -1, 1])
            verts = np.array([v1, v2, v3, v4])
            
            # Midpoints
            m12 = (v1 + v2) / 2
            m13 = (v1 + v3) / 2
            m14 = (v1 + v4) / 2
            m23 = (v2 + v3) / 2
            m24 = (v2 + v4) / 2
            m34 = (v3 + v4) / 2
            mids = np.array([m12, m13, m14, m23, m24, m34])
            
            # 4 random points near the origin or random
            extra = np.random.randn(4, 3) * 0.5
            
            x0 = np.vstack([verts, mids, extra])
        else:
            # Random points in a sphere of radius 2
            x0 = np.random.randn(N, D) * 2.0
            
        # Flatten
        x0_flat = x0.flatten()
        
        def objective(x_flat):
            pts = x_flat.reshape(N, D)
            # Center of mass to 0 to reduce degrees of freedom / drift
            pts = pts - np.mean(pts, axis=0)
            
            # Scale penalty to keep points in a reasonable range
            # Target RMS distance 1
            scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
            scale_penalty = (scale - 1.0) ** 2
            
            # Smooth ratio
            ratio = _smooth_ratio(pts, tau=0.5)
            
            return ratio + 10.0 * scale_penalty
        
        # Optimize
        try:
            res = minimize(objective, x0_flat, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6})
        except Exception:
            continue
            
        if res.success or res.fun < best_score:
            pts_opt = res.x.reshape(N, D)
            pts_opt = pts_opt - np.mean(pts_opt, axis=0)
            
            # Calculate actual score
            score = _compute_ratio(pts_opt)
            
            if score < best_score:
                best_score = score
                best_points = pts_opt
                
            # If we get a very good score, we might stop early?
            # Theoretical lower bound for 3D packing?
            # For N=14, it's hard to know. 1.0 is the seed.
            # If best_score < 1.1, we might be close to optimal.
            if best_score < 1.05:
                # We have improved significantly. Maybe try a few more to be sure.
                pass

    # Final fallback: if no optimization succeeded, return random or seed
    if best_points is None:
        best_points = np.random.randn(N, D)
        best_points = best_points - np.mean(best_points, axis=0)
        
    return best_points
