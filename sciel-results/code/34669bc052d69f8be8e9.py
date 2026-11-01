import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius in degrees for a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest neighbor. This is equal to the maximum circumradius of the facets
    of the convex hull of the points.
    """
    if len(points) < 3:
        return 180.0 # Degenerate case, shouldn't happen for N=50
        
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., coplanar points)
        # Estimate by max distance to centroid? No, just return a large value or use brute force on a grid?
        # For robustness, let's return a large number so this config is rejected.
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Calculate normal vector to the plane defined by p1, p2, p3
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        
        if norm < 1e-10:
            continue
            
        c = normal / norm
        
        # Ensure c is pointing outwards (away from origin)
        # The centroid of the triangle should be in the direction of c
        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
            
        # Calculate angular distance from c to any of the vertices
        # c is on the unit sphere, p1 is on the unit sphere
        dot_val = np.dot(c, p1)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        angle = np.arccos(dot_val)
        
        if angle > max_rad:
            max_rad = angle
            
    return np.degrees(max_rad)

def _objective(flat_x, beta, lambda_pen):
    """
    Objective function to minimize.
    Combines a soft-min of pairwise angular distances (to maximize min distance)
    and a penalty for points not being on the unit sphere.
    """
    X = flat_x.reshape(N, 3)
    
    # Penalty for unit sphere constraint
    norms_sq = np.sum(X * X, axis=1)
    penalty = lambda_pen * np.sum((norms_sq - 1.0) ** 2)
    
    # Compute pairwise dot products
    # X is N x 3
    # Dot product matrix D is N x N
    D = np.dot(X, X.T)
    
    # We only care about i < j
    # Clip to avoid numerical issues with arccos
    D = np.clip(D, -1.0, 1.0)
    
    # Compute angular distances
    # Only upper triangle to save computation, but for vectorization, full matrix is easier
    # However, arccos of diagonal is 0, which will dominate the sum if not handled.
    # We want min over i != j.
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # To avoid computing arccos on the whole matrix and then masking, 
    # we can just compute on the masked values.
    # But vectorizing arccos on a masked array is tricky in pure numpy without fancy indexing.
    # Let's just compute full, set diagonal to 1 (arccos(1)=0), and then ignore diagonal in sum.
    # Actually, we want to minimize the soft-min of the *non-zero* distances.
    # If we include 0s, the soft-min will be 0.
    # So we must exclude i=j.
    
    # Efficient way:
    # Get indices for i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    dots = D[i_idx, j_idx]
    
    # Clip again for safety
    dots = np.clip(dots, -1.0, 1.0)
    
    angles = np.arccos(dots)
    
    # Soft min: (1/beta) * log(sum(exp(-beta * angles)))
    # To avoid overflow/underflow in exp, use log-sum-exp trick
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    # Here x = -beta * angles
    x = -beta * angles
    max_x = np.max(x)
    if not np.isfinite(max_x):
        return penalty + 1e10 # Failure
        
    log_sum_exp = max_x + np.log(np.sum(np.exp(x - max_x)))
    
    soft_min = (1.0 / beta) * log_sum_exp
    
    # We want to MAXIMIZE the min distance, so we MINIMIZE -soft_min
    # Wait, soft_min is a lower bound on the min distance?
    # As beta -> inf, soft_min -> min(angles).
    # So minimizing soft_min is WRONG. We want to MAXIMIZE the min distance.
    # So we should MINIMIZE -soft_min.
    
    return -soft_min + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit
    
    best_X = None
    best_score = float('inf')
    
    # Parameters
    beta = 10.0
    lambda_pen = 100.0
    
    # Initial configurations
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    initial_configs = [X_fib]
    
    # Add some random configs
    rng = np.random.default_rng(42)
    for _ in range(4):
        X_rand = rng.normal(size=(N, 3))
        X_rand /= np.linalg.norm(X_rand, axis=1, keepdims=True)
        initial_configs.append(X_rand)
    
    for idx, X0 in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        x0 = X0.flatten()
        
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', args=(beta, lambda_pen),
                           options={'maxiter': 200, 'ftol': 1e-8})
        except Exception:
            continue
            
        if res.success or res.fun < best_score: # Check if better even if not "success"
            X_opt = res.x.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            X_opt = X_opt / norms
            
            score = _covering_radius(X_opt)
            
            if score < best_score:
                best_score = score
                best_X = X_opt
                
        # Optional: Polish with higher beta?
        # If we have time, we can run a second pass with higher beta on the best result found so far.
        
    # If we have time, do a final polish on the best configuration found
    if best_X is not None and time.time() < deadline - 10.0:
        x0 = best_X.flatten()
        beta_high = 50.0
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', args=(beta_high, lambda_pen),
                           options={'maxiter': 200, 'ftol': 1e-10})
            X_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            X_opt = X_opt / norms
            
            score = _covering_radius(X_opt)
            if score < best_score:
                best_X = X_opt
                best_score = score
        except Exception:
            pass
            
    if best_X is None:
        # Fallback to Fibonacci if optimization failed completely
        return X_fib
        
    return best_X
