"""Program to maximize the product of distances for 8 planar points with fixed diameter."""
import numpy as np
import time
from math import cos, sin, pi, log, sqrt
from scipy.optimize import minimize, differential_evolution

def _score(points):
    """Compute Delta / n^n where Delta is product of ordered-pair distances after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Diameter
    diam = np.max(dists)
    if diam == 0:
        return 0.0
    
    # Scale to diameter 2
    scale = 2.0 / diam
    dists_scaled = dists * scale
    
    # Product over ordered pairs (i != j)
    mask = ~np.eye(n, dtype=bool)
    # Avoid log(0) if points coincide
    if np.any(dists_scaled[mask] < 1e-12):
        return 0.0
        
    log_prod = np.sum(np.log(dists_scaled[mask]))
    score = np.exp(log_prod) / (n ** n)
    return score

def _neg_log_score_from_params(params):
    """
    Compute negative log score for a C4 symmetric configuration defined by:
    params = [r1, theta2, r2]
    Orbit 1: r1 * exp(i * (0, pi/2, pi, 3pi/2))
    Orbit 2: r2 * exp(i * (theta2, theta2 + pi/2, ...))
    """
    r1 = params[0]
    theta2 = params[1]
    r2 = params[2]
    
    # Generate points
    pts1 = np.array([
        [r1 * cos(0), r1 * sin(0)],
        [r1 * cos(pi/2), r1 * sin(pi/2)],
        [r1 * cos(pi), r1 * sin(pi)],
        [r1 * cos(3*pi/2), r1 * sin(3*pi/2)]
    ])
    
    pts2 = np.array([
        [r2 * cos(theta2), r2 * sin(theta2)],
        [r2 * cos(theta2 + pi/2), r2 * sin(theta2 + pi/2)],
        [r2 * cos(theta2 + pi), r2 * sin(theta2 + pi)],
        [r2 * cos(theta2 + 3*pi/2), r2 * sin(theta2 + 3*pi/2)]
    ])
    
    pts = np.vstack([pts1, pts2])
    
    # Calculate score
    n = 8
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam < 1e-12:
        return 1e10
        
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    
    # Check for zero distances
    if np.any(dists_scaled[mask] < 1e-12):
        return 1e10
        
    log_prod = np.sum(np.log(dists_scaled[mask]))
    neg_log_score = n * log(n) - log_prod
    return neg_log_score

def _two_squares_from_params(params):
    """
    Generate 8 points as two concentric, rotated regular squares.
    params = [r1, theta1, r2, theta2]
    Square 1: vertices at radius r1, starting angle theta1
    Square 2: vertices at radius r2, starting angle theta2
    """
    r1, theta1, r2, theta2 = params
    angles1 = theta1 + np.array([0, pi/2, pi, 3*pi/2])
    angles2 = theta2 + np.array([0, pi/2, pi, 3*pi/2])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    return np.vstack([pts1, pts2])

def _neg_log_score_two_squares(params):
    """Negative log score for two-squares configuration."""
    pts = _two_squares_from_params(params)
    n = 8
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam < 1e-12:
        return 1e10
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    if np.any(dists_scaled[mask] < 1e-12):
        return 1e10
    log_prod = np.sum(np.log(dists_scaled[mask]))
    return n * log(n) - log_prod

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 8
    best_score = 0.0
    best_pts = None
    
    # 1. Try C4 symmetric optimization (3 variables) - original approach
    try:
        res = minimize(_neg_log_score_from_params, 
                       x0=[1.0, pi/4, 1.0], 
                       method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 5000})
        
        if res.fun < 1e6:
            r1, theta2, r2 = res.x
            pts1 = np.array([
                [r1 * cos(0), r1 * sin(0)],
                [r1 * cos(pi/2), r1 * sin(pi/2)],
                [r1 * cos(pi), r1 * sin(pi)],
                [r1 * cos(3*pi/2), r1 * sin(3*pi/2)]
            ])
            pts2 = np.array([
                [r2 * cos(theta2), r2 * sin(theta2)],
                [r2 * cos(theta2 + pi/2), r2 * sin(theta2 + pi/2)],
                [r2 * cos(theta2 + pi), r2 * sin(theta2 + pi)],
                [r2 * cos(theta2 + 3*pi/2), r2 * sin(theta2 + 3*pi/2)]
            ])
            candidate_pts = np.vstack([pts1, pts2])
            sc = _score(candidate_pts)
            if sc > best_score:
                best_score = sc
                best_pts = candidate_pts
    except Exception as e:
        pass
    
    # 2. NEW: Two concentric rotated squares with independent orientations (4 params)
    # This is a different structural family: both squares can rotate independently.
    # Regular octagon corresponds to r1=r2, theta2-theta1=pi/4.
    # We search over r1, theta1, r2, theta2.
    
    if time.time() < deadline - 30.0:
        try:
            # Bounds: radii in [0.1, 2.0], angles in [0, 2*pi)
            bounds = [(0.1, 2.0), (0.0, 2.0*pi), (0.1, 2.0), (0.0, 2.0*pi)]
            
            de_res = differential_evolution(_neg_log_score_two_squares, bounds,
                                            maxiter=100, popsize=20, seed=42,
                                            tol=1e-8, polish=False, init='sobol')
            
            if de_res.fun < 1e6:
                candidate_pts = _two_squares_from_params(de_res.x)
                # Polish with Nelder-Mead
                nm_res = minimize(_neg_log_score_two_squares, de_res.x,
                                  method='Nelder-Mead',
                                  options={'xatol': 1e-10, 'fatol': 1e-10,
                                           'maxiter': 10000, 'maxfev': 10000})
                if nm_res.fun < 1e6 and nm_res.fun <= de_res.fun + 1e-6:
                    candidate_pts = _two_squares_from_params(nm_res.x)
                
                sc = _score(candidate_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = candidate_pts
        except Exception as e:
            pass
    
    # 3. Try broader 16D DE if time permits
    if time.time() < deadline - 20.0:
        bounds_16d = [(-2.0, 2.0)] * 16
        
        def _neg_log_score_16d(pts):
            n = 8
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            diam = np.max(dists)
            if diam < 1e-12:
                return 1e10
            scale = 2.0 / diam
            dists_scaled = dists * scale
            mask = ~np.eye(n, dtype=bool)
            if np.any(dists_scaled[mask] < 1e-12):
                return 1e10
            log_prod = np.sum(np.log(dists_scaled[mask]))
            return n * log(n) - log_prod
        
        def _de_objective(x_flat):
            if time.time() > deadline - 5.0:
                return 1e10
            pts = x_flat.reshape(8, 2)
            return _neg_log_score_16d(pts)
        
        try:
            de_res = differential_evolution(_de_objective, bounds_16d, maxiter=50, popsize=15, seed=42, tol=1e-6, polish=False)
            
            if de_res.fun < 1e6:
                candidate_pts = de_res.x.reshape(8, 2)
                
                if time.time() < deadline - 5.0:
                    slsqp_res = minimize(_de_objective, candidate_pts.flatten(), method='SLSQP', 
                                         options={'maxiter': 100, 'ftol': 1e-12})
                    if slsqp_res.fun < 1e6:
                        candidate_pts = slsqp_res.x.reshape(8, 2)
                    
                    sc = _score(candidate_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = candidate_pts
        except Exception as e:
            pass

    # 4. Fallback: Regular Octagon
    if best_pts is None:
        best_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
        best_score = _score(best_pts)
        
    return best_pts.tolist()
