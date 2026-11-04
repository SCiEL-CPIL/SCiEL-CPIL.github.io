"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize
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

def _smooth_min(values, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_val = np.min(values)
    shifted = -tau * (values - min_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_val - (1.0 / tau) * log_sum

def _objective(params, tau):
    """
    Objective function for optimization.
    params: 2N values representing x and y coordinates.
    """
    pts = params.reshape(-1, 2)
    
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
    
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _generate_initial_params():
    """Generate initial parameters for a regular 13-gon."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y)).flatten()

def _differential_evolution_step(objective, bounds, initial, tau, time_limit, start_time):
    """Run differential evolution with time constraint."""
    def bounded_objective(params):
        if (time.time() - start_time) > time_limit:
            return 1e9
        return objective(params, tau)
    
    try:
        result = differential_evolution(
            bounded_objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            popsize=15,
            init='sobol',
            polish=False,
            callback=lambda x, convergence: (time.time() - start_time) < time_limit
        )
        return result
    except Exception:
        return None

def _nelder_mead_polish(objective, x0, tau, time_limit, start_time):
    """Polish the result using Nelder-Mead."""
    def bounded_objective(params):
        if (time.time() - start_time) > time_limit:
            return 1e9
        return objective(params, tau)
    
    try:
        result = minimize(
            bounded_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        return result
    except Exception:
        return None

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_params = None
    
    # Bounds for each coordinate: roughly [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # Annealing schedule for tau
    taus = [20, 50, 100, 200, 500]
    
    # Multiple restarts
    num_restarts = 8
    
    for restart in range(num_restarts):
        if (time.time() - start_time) > time_limit - 15.0:
            break
            
        # Generate initial guess
        if restart == 0:
            current_params = _generate_initial_params()
        elif best_params is not None and rng.random() < 0.7:
            current_params = best_params + rng.normal(0, 0.05, 2 * N)
        else:
            # Random points in a disk
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0.5, 1.5, N)
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            current_params = np.column_stack((x, y)).flatten()
        
        for tau in taus:
            if (time.time() - start_time) > time_limit - 10.0:
                break
            
            # Differential evolution step
            de_result = _differential_evolution_step(_objective, bounds, current_params, tau, time_limit - 5.0, start_time)
            
            if de_result is not None and de_result.fun < 1e8:
                candidate_params = de_result.x
                pts = candidate_params.reshape(-1, 2)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = candidate_params.copy()
                    current_params = candidate_params.copy()
            
            # Nelder-Mead polish
            if (time.time() - start_time) > time_limit - 5.0:
                break
                
            nm_result = _nelder_mead_polish(_objective, current_params, tau, time_limit - 2.0, start_time)
            
            if nm_result is not None and nm_result.fun < 1e8:
                candidate_params = nm_result.x
                pts = candidate_params.reshape(-1, 2)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = candidate_params.copy()
                    current_params = candidate_params.copy()
    
    # If no valid solution found, return a default regular polygon
    if best_params is None:
        best_params = _generate_initial_params()
    
    # Construct final points
    final_pts = best_params.reshape(-1, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
