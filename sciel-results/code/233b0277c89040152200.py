import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # We optimize the angular gaps between consecutive points on a circle.
    # There are N gaps, but they sum to 2*pi, so we have N-1 free variables.
    
    # Pre-compute indices for all triples (i, j, k) with i < j < k
    # This is constant for a given N
    i_idx, j_idx, k_idx = np.indices((N, N, N)).reshape(3, -1).T
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
            
        # Calculate cumulative angles
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Vectorized calculation of min triangle area
        a = phi[j_idx] - phi[i_idx]
        b = phi[k_idx] - phi[j_idx]
        c = 2 * np.pi - a - b
        
        # Area factor
        areas = np.sin(a) + np.sin(b) + np.sin(c)
        
        # The minimum area is the minimum of these values
        min_num = np.min(areas)
        
        # Hull area is proportional to sum of sin(deltas)
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 0.0
            
        ratio = min_num / den
        return -ratio

    # Initial guess: uniform distribution
    delta_init = np.full(N-1, 2 * np.pi / N)
    
    # Bounds for each gap
    bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
    
    best_val = -np.inf
    best_x = delta_init
    
    # Multi-start optimization to escape local minima
    rng = np.random.default_rng(42)
    
    # Start with uniform
    res = minimize(
        objective, 
        delta_init, 
        method='Nelder-Mead', 
        bounds=bounds,
        options={
            'xatol': 1e-10, 
            'fatol': 1e-10, 
            'maxiter': 2000, 
            'adaptive': True
        }
    )
    if res.fun < best_val:
        best_val = res.fun
        best_x = res.x
        
    # Try random restarts
    for _ in range(20):
        # Perturb uniform gaps
        noise = rng.normal(0, 0.1, N-1)
        x0 = delta_init + noise
        # Ensure positive
        x0 = np.maximum(x0, 1e-5)
        # Normalize to sum to < 2*pi
        s = np.sum(x0)
        if s >= 2 * np.pi:
            x0 = x0 * (2 * np.pi * 0.9) / s
            
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={
                'xatol': 1e-10, 
                'fatol': 1e-10, 
                'maxiter': 1000, 
                'adaptive': True
            }
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    # Use the best found x
    opt_deltas = np.empty(N)
    opt_deltas[:N-1] = best_x
    opt_deltas[N-1] = 2 * np.pi - np.sum(best_x)
    
    # Fallback if something went wrong
    if np.any(opt_deltas <= 0):
        opt_deltas = np.full(N, 2 * np.pi / N)

    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
