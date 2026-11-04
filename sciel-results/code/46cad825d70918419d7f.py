"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices for speed
_triple_indices = np.array(list(combinations(range(N), 3)))

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    # Extract indices
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-9:
        return 0.0
        
    return min_area / hull_area

def _logsumexp_min(ratios, tau):
    """
    Smooth approximation of the minimum of ratios using LogSumExp.
    min(x) = -lim_{tau->inf} (1/tau) log(sum(exp(-tau * x)))
    """
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    # shifted <= 0, max is 0
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_r - (1.0 / tau) * log_sum

def _points_from_log_radii(log_radii, base_angles):
    """
    Reconstruct points from log-radii and base angles.
    We fix the first two points to define the scale and rotation implicitly?
    No, the task says "normalized to unit area" but the interface returns points.
    The score is invariant to scaling. 
    To make the optimization well-posed, we can fix the scale by setting the 
    average radius or just let the optimizer find the shape.
    However, the ratio is scale-invariant. 
    Let's fix the center at (0,0) and use the radii.
    The angles are fixed to the regular polygon angles.
    """
    radii = np.exp(log_radii)
    x = radii * np.cos(base_angles)
    y = radii * np.sin(base_angles)
    return np.column_stack((x, y))

def _objective(log_radii, tau, base_angles):
    """
    Objective function for SLSQP.
    We minimize the negative of the smoothed min ratio.
    """
    pts = _points_from_log_radii(log_radii, base_angles)
    
    # Check for degeneracy
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 1e9
        
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-9:
        return 1e9
        
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    
    approx_min = _logsumexp_min(ratios, tau)
    return -approx_min

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Base angles for a regular 13-gon
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    best_score = -np.inf
    best_log_radii = None
    
    # Initial guess: Regular polygon (all radii = 1)
    # log(1) = 0
    init_log_radii = np.zeros(N)
    
    # Annealing schedule for tau
    # Start with small tau (smooth) and increase (sharp)
    taus = [10, 50, 200, 1000, 5000]
    
    # Multiple restarts with random perturbations in log-space
    num_restarts = 10
    
    for restart in range(num_restarts):
        if (time.time() - start_time) > time_limit - 10.0:
            break
            
        # Generate initial guess
        if restart == 0:
            current_log_radii = init_log_radii.copy()
        else:
            # Perturb the best found so far or generate new
            if best_log_radii is not None and rng.random() < 0.7:
                current_log_radii = best_log_radii + rng.normal(0, 0.1, N)
            else:
                # Random radii between 0.5 and 1.5
                current_log_radii = np.log(rng.uniform(0.5, 1.5, N))
        
        for tau in taus:
            if (time.time() - start_time) > time_limit - 5.0:
                break
                
            try:
                # Bounds for log_radii: exp(-2) ~ 0.13, exp(2) ~ 7.4
                # This is a wide range, allowing significant deformation
                bounds = [(-2.0, 2.0)] * N
                
                res = minimize(
                    _objective, 
                    current_log_radii, 
                    args=(tau, base_angles),
                    method='SLSQP', 
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'iprint': 0}
                )
                
                # Check if result is valid and better
                if res.fun < 1e8: # Not a penalty value
                    new_log_radii = res.x
                    pts = _points_from_log_radii(new_log_radii, base_angles)
                    score = _compute_min_ratio(pts)
                    
                    if score > best_score:
                        best_score = score
                        best_log_radii = new_log_radii.copy()
                        current_log_radii = new_log_radii.copy()
                        
            except Exception as e:
                pass

    # If no valid solution found, return a default regular polygon
    if best_log_radii is None:
        best_log_radii = init_log_radii
        
    # Construct final points
    final_pts = _points_from_log_radii(best_log_radii, base_angles)
    
    return final_pts

# EVOLVE-BLOCK-END
