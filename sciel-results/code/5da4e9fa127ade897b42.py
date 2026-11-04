import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # Pre-compute indices for all triples to vectorize area calculation
    # i < j < k
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    # Filter for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    def objective(x):
        # x: array of first N-1 gaps
        # Check for invalid gaps (must be positive)
        if np.any(x <= 0):
            return 0.0
            
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        
        # Check if the last gap is valid
        if deltas[N-1] <= 0:
            return 0.0
            
        # Calculate cumulative angles (starting at 0)
        # phi[i] is the angle of point i
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Vectorized calculation of triangle areas
        # a = phi[j] - phi[i]
        # b = phi[k] - phi[j]
        # c = 2*pi - a - b
        # Note: phi is sorted, so a, b are positive.
        # c is the remaining arc.
        a = phi[j_idx] - phi[i_idx]
        b = phi[k_idx] - phi[j_idx]
        c = 2 * np.pi - a - b
        
        # Area is proportional to sin(a) + sin(b) + sin(c)
        # We want to maximize the minimum of this value
        areas = np.sin(a) + np.sin(b) + np.sin(c)
        min_num = np.min(areas)
        
        # Calculate hull area denominator
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 0.0
            
        ratio = min_num / den
        return -ratio # We minimize the negative ratio to maximize the ratio

    # Bounds for each gap
    # They must be positive and sum to 2*pi
    # A simple bound is (1e-5, 2*pi - 1e-5)
    bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
    
    # Initial guess: uniform distribution
    delta_init = np.full(N-1, 2 * np.pi / N)
    
    # Use Powell method which is robust for non-smooth objectives (due to min)
    # and doesn't require gradients.
    # We will do a few restarts to avoid local minima.
    
    best_result = None
    best_score = float('inf')
    
    # 1. Start from uniform
    result = minimize(
        objective, 
        delta_init, 
        method='Powell', 
        bounds=bounds,
        options={
            'xatol': 1e-8, 
            'fatol': 1e-8, 
            'maxiter': 1000, 
            'maxfev': 5000
        }
    )
    if result.success or result.fun < best_score:
        best_result = result
        best_score = result.fun
        
    # 2. Try a few random perturbations of the uniform start
    np.random.seed(42)
    for _ in range(5):
        # Perturb the gaps slightly
        perturbation = np.random.uniform(-0.2, 0.2, N-1)
        # Ensure they don't go too far
        start = delta_init + perturbation
        # Clip to bounds roughly to avoid immediate invalidity
        start = np.clip(start, 1e-5, 2*np.pi - 1e-5)
        
        result = minimize(
            objective, 
            start, 
            method='Powell', 
            bounds=bounds,
            options={
                'xatol': 1e-8, 
                'fatol': 1e-8, 
                'maxiter': 1000, 
                'maxfev': 5000
            }
        )
        if result.success or result.fun < best_score:
            best_result = result
            best_score = result.fun
            
    if best_result is not None:
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = best_result.x
        opt_deltas[N-1] = 2 * np.pi - np.sum(best_result.x)
    else:
        # Fallback to uniform if optimization fails
        opt_deltas = np.full(N, 2 * np.pi / N)
        
    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
