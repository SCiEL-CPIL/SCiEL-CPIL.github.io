import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # Pre-compute indices for all triples
    indices = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                indices.append((i, j, k))
    indices = np.array(indices)
    i_idx = indices[:, 0]
    j_idx = indices[:, 1]
    k_idx = indices[:, 2]

    def objective(x):
        # x: array of first N-1 gaps
        if np.any(x <= 0):
            return 1e10 # Large penalty
        
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        
        if deltas[N-1] <= 0:
            return 1e10

        # Cumulative angles
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Calculate arc differences for all triples
        # a = phi[j] - phi[i]
        # b = phi[k] - phi[j]
        # c = 2*pi - (a+b)
        a = phi[j_idx] - phi[i_idx]
        b = phi[k_idx] - phi[j_idx]
        c = 2 * np.pi - a - b
        
        # Area numerator (2 * Area)
        areas_num = np.sin(a) + np.sin(b) + np.sin(c)
        
        # Handle potential numerical issues where area might be slightly negative due to precision
        # though for convex ordering it should be positive
        areas_num = np.maximum(areas_num, 1e-9)
        
        min_num = np.min(areas_num)
        
        # Hull area denominator (2 * Hull Area)
        den = np.sum(np.sin(deltas))
        
        if den <= 1e-9:
            return 1e10
            
        ratio = min_num / den
        return -ratio

    # Bounds
    bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
    
    best_ratio = -np.inf
    best_x = np.full(N-1, 2 * np.pi / N)
    
    # Multiple restarts
    # Start with uniform
    restarts = [np.full(N-1, 2 * np.pi / N)]
    
    # Add some random perturbations
    np.random.seed(42)
    for _ in range(5):
        # Perturb uniform
        pert = np.random.normal(0, 0.1, N-1)
        x_start = np.full(N-1, 2 * np.pi / N) + pert
        # Ensure positive
        x_start = np.maximum(x_start, 1e-5)
        # Normalize sum? No, the last gap adjusts.
        # But we need sum(x_start) < 2*pi
        s = np.sum(x_start)
        if s > 2 * np.pi - 1e-5:
             x_start = x_start * (2 * np.pi - 1e-5) / s
        restarts.append(x_start)
        
    for x0 in restarts:
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={
                'maxiter': 1000, 
                'ftol': 1e-12,
                'gtol': 1e-8
            }
        )
        
        if result.success:
            curr_ratio = -result.fun
            if curr_ratio > best_ratio:
                best_ratio = curr_ratio
                best_x = result.x
    
    # Fallback if all fail
    if not np.isfinite(best_ratio):
        best_x = np.full(N-1, 2 * np.pi / N)

    opt_deltas = np.empty(N)
    opt_deltas[:N-1] = best_x
    opt_deltas[N-1] = 2 * np.pi - np.sum(best_x)
    
    # Construct points
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
