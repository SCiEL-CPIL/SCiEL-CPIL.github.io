"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 135.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _soft_min_triangle_area(points, beta=50.0):
    """
    Approximate the minimum triangle area using log-sum-exp (soft-min).
    This makes the objective differentiable for Nelder-Mead.
    """
    n = len(points)
    areas = []
    x = points[:, 0]
    y = points[:, 1]
    # We accumulate the log-sum-exp terms directly to avoid storing all areas
    # and to handle potential underflow/overflow better, though for small N
    # storing is fine. Given N=14, 364 triangles, storing is trivial.
    
    log_sum_exp_val = -np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            abs_cross = np.abs(cross)
            # Area = 0.5 * abs_cross
            # We want soft_min of areas.
            # soft_min(A) = -1/beta * log(sum(exp(-beta * A)))
            # Let's compute S = sum(exp(-beta * 0.5 * abs_cross))
            # Then soft_min = -1/beta * log(S)
            
            # To prevent overflow in exp, we can shift by max, but 
            # since we are summing exp(-beta * A), and A >= 0, 
            # the largest term is for the smallest A.
            # If A is very small, exp(0) = 1. If A is large, exp(-large) -> 0.
            # So no overflow risk for positive beta and positive A.
            
            terms = np.exp(-beta * 0.5 * abs_cross)
            S = np.sum(terms)
            
            # We need the global soft min over ALL triangles.
            # soft_min_global = -1/beta * log( sum_{all} exp(-beta * A_k) )
            # We are summing exp(-beta * A_k) for each group.
            # Let's just collect all areas first, it's only 364 values.
            areas.extend(abs_cross * 0.5)
            
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    
    # Calculate soft min
    # min_A = np.min(areas)
    # soft_min = min_A - (1/beta) * log(sum(exp(-beta * (areas - min_A))))
    # This is numerically stable.
    min_A = np.min(areas)
    shifted = areas - min_A
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_A - (1.0 / beta) * log_sum
    
    return soft_min


def _soft_score(params):
    """
    Objective for Nelder-Mead.
    params: [r0, theta0, r1, theta1, r2, theta2]
    Ring 0: 7 points, radius r0, angles theta0 + 2*pi*k/7
    Ring 1: 7 points, radius r1, angles theta1 + 2*pi*k/7
    """
    r0, theta0, r1, theta1, r2, theta2 = params
    
    # Ensure radii are positive
    if r0 <= 0 or r1 <= 0 or r2 <= 0:
        return 1e10
        
    # Construct points
    angles0 = theta0 + 2 * np.pi * np.arange(7) / 7.0
    angles1 = theta1 + 2 * np.pi * np.arange(7) / 7.0
    
    pts0 = np.column_stack([r0 * np.cos(angles0), r0 * np.sin(angles0)])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    points = np.vstack([pts0, pts1])
    
    # Normalize to unit area
    hull_a = _hull_area(points)
    if hull_a <= 0:
        return 1e10
        
    points_normalized = points / np.sqrt(hull_a)
    
    # Use soft min area for differentiability
    # We return negative soft score because minimize finds minimum
    soft_min_area = _soft_min_triangle_area(points_normalized, beta=50.0)
    score = soft_min_area / 1.0  # Hull area is 1
    
    return -score


def _generalized_pattern_search(points, max_iter=200, delta_init=0.1, delta_min=1e-5):
    """
    Local search on the exact non-smooth objective using generalized pattern search.
    Tries expanding and shrinking steps in coordinate directions.
    """
    current_pts = points.copy()
    current_score = _score(current_pts)
    delta = delta_init
    
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
            
        improved = False
        # Directions: +/- e_i for all 2N coordinates
        for i in range(2 * N):
            for direction in [1.0, -1.0]:
                trial_pts = current_pts.copy()
                idx1, idx2 = divmod(i, 2)
                trial_pts[idx1, idx2] += direction * delta
                
                # Ensure points don't go too far out or collapse
                if np.any(np.linalg.norm(trial_pts, axis=1) < 1e-6):
                    continue
                    
                trial_score = _score(trial_pts)
                
                if trial_score > current_score:
                    current_pts = trial_pts
                    current_score = trial_score
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            # Expand step size slightly to accelerate
            delta = min(delta * 1.5, 0.5)
        else:
            # Shrink step size
            delta *= 0.5
            if delta < delta_min:
                break
                
    return current_pts, current_score


def construct():
    start_time = time.time()
    
    # Initial guess: two concentric heptagons
    # Regular heptagon area is approx 2.736 * r^2.
    # If we have two rings, we need to balance radii.
    # Let's start with r0=0.5, r1=1.0
    x0 = [0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
    
    best_pts = None
    best_score = 0.0
    
    # Run Nelder-Mead on the smooth surrogate
    try:
        result = minimize(
            _soft_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if result.x is not None:
            r0, theta0, r1, theta1, r2, theta2 = result.x
            if r0 > 0 and r1 > 0:
                angles0 = theta0 + 2 * np.pi * np.arange(7) / 7.0
                angles1 = theta1 + 2 * np.pi * np.arange(7) / 7.0
                
                pts0 = np.column_stack([r0 * np.cos(angles0), r0 * np.sin(angles0)])
                pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
                
                cand_pts = np.vstack([pts0, pts1])
                # Normalize
                hull_a = _hull_area(cand_pts)
                if hull_a > 0:
                    cand_pts = cand_pts / np.sqrt(hull_a)
                    cand_score = _score(cand_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts
    except Exception:
        pass
    
    # If NM failed or didn't find anything, use a default geometric config
    if best_pts is None:
        angles0 = 2 * np.pi * np.arange(7) / 7.0
        angles1 = angles0 + np.pi / 7.0  # Interleaved
        pts0 = np.column_stack([0.5 * np.cos(angles0), 0.5 * np.sin(angles0)])
        pts1 = np.column_stack([1.0 * np.cos(angles1), 1.0 * np.sin(angles1)])
        best_pts = np.vstack([pts0, pts1])
        hull_a = _hull_area(best_pts)
        if hull_a > 0:
            best_pts = best_pts / np.sqrt(hull_a)
        best_score = _score(best_pts)
    
    # Polish with Generalized Pattern Search on exact objective
    if time.time() - start_time < DEADLINE - 10:
        try:
            polished_pts, polished_score = _generalized_pattern_search(
                best_pts, 
                max_iter=500, 
                delta_init=0.1, 
                delta_min=1e-6
            )
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts
        except Exception:
            pass
            
    return best_pts

# EVOLVE-BLOCK-END
