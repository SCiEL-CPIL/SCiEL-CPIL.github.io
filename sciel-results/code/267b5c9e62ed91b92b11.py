import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 14

def construct():
    # We optimize the angular gaps between consecutive points on a circle.
    # There are N gaps, but they sum to 2*pi, so we have N-1 free variables.
    
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
        
        # Calculate the minimum triangle area numerator
        # Area ~ sin(a) + sin(b) + sin(c) where a,b,c are arcs
        min_num = float('inf')
        
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    # Arcs between i-j, j-k, k-i
                    a = phi[j] - phi[i]
                    b = phi[k] - phi[j]
                    c = 2 * np.pi - a - b
                    
                    val = np.sin(a) + np.sin(b) + np.sin(c)
                    if val < min_num:
                        min_num = val
        
        # Calculate hull area denominator
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 0.0
            
        ratio = min_num / den
        return -ratio # We minimize the negative ratio to maximize the ratio

    # Bounds for each gap
    bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
    
    # Stage 1: Differential Evolution for global search
    de_result = differential_evolution(
        objective, 
        bounds, 
        seed=42,
        maxiter=100,
        tol=1e-6,
        popsize=10
    )
    
    best_x = de_result.x
    best_val = -de_result.fun
    
    # Stage 2: Refine with Powell
    result = minimize(
        objective, 
        best_x, 
        method='Powell',
        bounds=bounds,
        options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-10}
    )
    
    if result.success and -result.fun > best_val:
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = result.x
        opt_deltas[N-1] = 2 * np.pi - np.sum(result.x)
    else:
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = best_x
        opt_deltas[N-1] = 2 * np.pi - np.sum(best_x)
        
    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
