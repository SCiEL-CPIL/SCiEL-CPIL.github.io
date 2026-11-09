# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _calculate_min_score(poly):
    """
    Calculate the minimum vertex score for a given polygon.
    Returns 0.0 if invalid.
    """
    n = len(poly)
    if n != 20:
        return 0.0
    
    # Check for repeated points
    dists_matrix = np.linalg.norm(poly[:, np.newaxis, :] - poly[np.newaxis, :, :], axis=2)
    # Off-diagonal elements should be > 0
    np.fill_diagonal(dists_matrix, np.inf)
    if np.min(dists_matrix) < 1e-9:
        return 0.0

    # Check convexity
    # Compute cross products of consecutive edges
    turns = np.zeros(n)
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i+1)%n]
        p2 = poly[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        turns[i] = v1[0]*v2[1] - v1[1]*v2[0]
        
    if np.any(np.isclose(turns, 0, atol=1e-8)):
        return 0.0
    if not (np.all(turns > 0) or np.all(turns < 0)):
        return 0.0
        
    # Normalize
    centroid = np.mean(poly, axis=0)
    poly_norm = poly - centroid
    dists_to_centroid = np.linalg.norm(poly_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    poly_norm = poly_norm / mean_dist
    
    # Recalculate distances for normalized poly
    dists_matrix_norm = np.linalg.norm(poly_norm[:, np.newaxis, :] - poly_norm[np.newaxis, :, :], axis=2)
    
    min_score = 1.0
    for i in range(n):
        d = dists_matrix_norm[i, 1:] # Exclude self
        d = np.sort(d)
        
        if d[0] < 0.05:
            return 0.0
            
        gaps = np.diff(d)
        k = np.argmin(gaps)
        D = (d[k] + d[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        if k == 0:
            sel = d[0:4]
        elif k == 17:
            sel = d[15:19]
        else:
            sel = d[k-1:k+3]
            
        # Handle division by zero or negative (shouldn't happen)
        if np.any(sel <= 0) or D <= 0:
            return 0.0
            
        ratios = np.maximum(sel/D, D/sel)
        raw = np.mean(ratios)
        score = min(1.0, 1.0/raw)
        if score < min_score:
            min_score = score
            if min_score < 1e-9:
                return 0.0
                
    return min_score

def _objective(params):
    """
    Objective function for optimization.
    params: array of 20 radial distances.
    Returns negative of the score.
    """
    r = params
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    poly = np.stack([x, y], axis=1)
    
    score = _calculate_min_score(poly)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that maximizes the minimum vertex score.
    Uses optimization on radial parameters of a star-shaped polygon.
    """
    n = 20
    start_time = time.time()
    time_limit = 580.0 # Safety margin
    
    # Initial guess: regular polygon
    r0 = np.ones(n)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    poly0 = np.stack([r0 * np.cos(angles), r0 * np.sin(angles)], axis=1)
    
    # Try a few different initializations and optimizations
    best_poly = poly0
    best_score = _calculate_min_score(poly0)
    
    # If regular polygon is already good, try to improve
    # Use Nelder-Mead
    options = {
        'maxiter': 2000,
        'xatol': 1e-6,
        'fatol': 1e-6,
        'adaptive': True
    }
    
    # We can try multiple random restarts if time permits
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if i == 0:
            r_init = r0
        else:
            # Perturb initial radii
            r_init = r0 * (1.0 + 0.1 * np.random.randn(n))
            # Ensure positive radii
            r_init = np.abs(r_init) + 0.1
        
        try:
            res = minimize(_objective, r_init, method='Nelder-Mead', options=options)
            if res.fun < -best_score: # i.e. -score < -best_score => score > best_score
                best_score = -res.fun
                r_opt = res.x
                poly_opt = np.stack([r_opt * np.cos(angles), r_opt * np.sin(angles)], axis=1)
                # Verify score
                actual_score = _calculate_min_score(poly_opt)
                if actual_score > best_score:
                    best_score = actual_score
                    best_poly = poly_opt
        except Exception as e:
            pass

    # If we have time, do a finer local search on the best found
    if time.time() - start_time < time_limit - 10:
        r_best = np.linalg.norm(best_poly, axis=1)
        # Simple hill climbing
        step_size = 0.01
        for _ in range(500):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                for delta in [-step_size, step_size]:
                    r_test = r_best.copy()
                    r_test[i] += delta
                    if r_test[i] < 0.1: continue
                    poly_test = np.stack([r_test * np.cos(angles), r_test * np.sin(angles)], axis=1)
                    score_test = _calculate_min_score(poly_test)
                    if score_test > best_score:
                        best_score = score_test
                        r_best = r_test
                        best_poly = poly_test
                        improved = True
            if not improved:
                step_size *= 0.5
                if step_size < 1e-5:
                    break

    return best_poly

# EVOLVE-BLOCK-END
