"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import random

def _score(points):
    """Compute the score: product of all pairwise distances (ordered) / 8^8, after scaling to diameter 2."""
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Diameter is max pairwise distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    # Scale so diameter = 2
    scale = 2.0 / diam
    dists_scaled = dists * scale
    # Product over ordered pairs (i != j)
    # We want product of dists[i,j] for all i != j
    # This is equivalent to (product of all unordered pair distances)^2
    # But let's compute directly: product of all off-diagonal entries
    mask = ~np.eye(n, dtype=bool)
    prod = np.prod(dists_scaled[mask])
    score = prod / (n ** n)
    return score

def _log_score(points):
    """Compute log of the score for numerical stability."""
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    diam = np.max(dists)
    if diam < 1e-15:
        return -np.inf
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    # log of product = sum of logs
    log_prod = np.sum(np.log(dists_scaled[mask] + 1e-300))
    log_score = log_prod - n * np.log(n)
    return log_score

def _regular_octagon():
    from math import cos, sin, pi
    return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

def _symmetric_config(params):
    """
    Generate 8 points from 4 parameters using conjugate symmetry.
    params = [r1, theta1, r2, theta2, r3, theta3, r4, theta4]
    Points are: r_k * exp(i * theta_k) and their conjugates, for k=1..4
    But we need 8 distinct points, so we use 4 complex numbers and their conjugates.
    Actually, to get 8 points with real-coefficient polynomial, we can use:
    4 pairs of conjugates: z1, conj(z1), z2, conj(z2), z3, conj(z3), z4, conj(z4)
    where zk = rk * exp(i * theta_k)
    """
    pts = []
    for k in range(4):
        r = params[2*k]
        theta = params[2*k+1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts.append([x, y])
        pts.append([x, -y])  # conjugate
    return pts

def _objective(params):
    """Negative log score for minimization."""
    pts = _symmetric_config(params)
    ls = _log_score(pts)
    if not np.isfinite(ls):
        return 1e10
    return -ls

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with regular octagon
    best_pts = _regular_octagon()
    best_score = _score(best_pts)
    best_log_score = _log_score(best_pts)
    
    # Try basin hopping style optimization in the symmetric parameter space
    # Parameters: [r1, theta1, r2, theta2, r3, theta3, r4, theta4]
    # Regular octagon in this parameterization:
    # Angles: 0, pi/4, pi/2, 3pi/4 (and their negatives)
    # r = 1 for all
    # But wait: regular octagon has angles 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
    # Conjugate pairs: (0, pi), (pi/4, 7pi/4), (pi/2, 3pi/2), (3pi/4, 5pi/4)
    # So params: [1, 0, 1, pi/4, 1, pi/2, 1, 3pi/4]
    
    from math import pi
    initial_params = [1.0, 0.0, 1.0, pi/4, 1.0, pi/2, 1.0, 3*pi/4]
    
    # Use multiple restarts with random perturbations
    random.seed(42)
    np.random.seed(42)
    
    best_params = initial_params[:]
    best_obj = _objective(initial_params)
    
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline and n_restarts < max_restarts:
        n_restarts += 1
        # Random perturbation of initial params
        if n_restarts == 1:
            params = initial_params[:]
        else:
            params = [p + random.gauss(0, 0.3) for p in initial_params]
        
        # Local optimization using BFGS
        try:
            result = minimize(_objective, params, method='BFGS', 
                             options={'maxiter': 500, 'gtol': 1e-10})
            if result.fun < best_obj and np.isfinite(result.fun):
                best_obj = result.fun
                best_params = result.x[:]
                pts = _symmetric_config(best_params)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts
                    best_log_score = _log_score(pts)
        except:
            pass
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Also try a few non-symmetric configurations with direct optimization
    # to see if breaking symmetry helps
    if time.time() < deadline - 10:
        # Direct 16D optimization with several restarts
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            # Start from regular octagon with small random perturbation
            pts = np.array(_regular_octagon(), dtype=np.float64)
            pts = pts + np.random.randn(8, 2) * 0.1
            # Flatten to 16D
            flat = pts.flatten()
            
            def obj_16d(x):
                pts_x = x.reshape(8, 2)
                ls = _log_score(pts_x)
                if not np.isfinite(ls):
                    return 1e10
                return -ls
            
            try:
                result = minimize(obj_16d, flat, method='BFGS',
                                 options={'maxiter': 300, 'gtol': 1e-10})
                if result.fun < best_obj and np.isfinite(result.fun):
                    pts_x = result.x.reshape(8, 2)
                    s = _score(pts_x)
                    if s > best_score:
                        best_score = s
                        best_pts = pts_x.tolist()
                        best_log_score = _log_score(pts_x)
                        best_obj = result.fun
            except:
                pass
    
    # Return best points found
    return best_pts
# EVOLVE-BLOCK-END
