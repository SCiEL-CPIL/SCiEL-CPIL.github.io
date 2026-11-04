import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 14

def construct():
    # We optimize the angular gaps between consecutive points on a circle.
    # There are N gaps, but they sum to 2*pi, so we have N-1 free variables.
    
    # Initial guess: uniform distribution
    delta_init = np.full(N-1, 2 * np.pi / N)
    
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
                    # Note: phi is in [0, 2*pi)
                    # a = phi[j] - phi[i]
                    # b = phi[k] - phi[j]
                    # c = 2*pi - (a + b)
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
    # They must be positive and sum to 2*pi
    # To ensure the last gap is also positive, we constrain the sum of the first N-1 to be less than 2*pi.
    # A safe lower bound is small positive. Upper bound should leave room for the last gap.
    # Let's use (1e-5, 2*pi - 1e-5) but DE handles bounds well.
    # However, if we set upper bound to 2*pi, the sum constraint is handled by the last variable calculation,
    # but DE doesn't know about the sum constraint directly.
    # We can just bound each variable to (1e-5, 2*pi - 1e-5).
    # The objective returns 0 if the last gap is invalid, which guides the search.
    
    bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
    
    # Use Differential Evolution for global optimization
    # DE is robust for non-convex problems
    result_de = differential_evolution(
        objective, 
        bounds, 
        seed=42,
        maxiter=200,
        tol=1e-8,
        polish=False  # We will do our own polish
    )
    
    best_x = result_de.x
    best_val = result_de.fun
    
    # Refine with a local optimizer (Powell is good for non-smooth or mildly smooth)
    # We use the best result from DE as the starting point
    try:
        result_local = minimize(
            objective, 
            best_x, 
            method='Powell',
            bounds=bounds,
            options={
                'maxiter': 1000, 
                'xtol': 1e-12, 
                'ftol': 1e-12
            }
        )
        if result_local.fun < best_val:
            best_x = result_local.x
            best_val = result_local.fun
    except Exception:
        pass
        
    # Construct points on the unit circle
    opt_deltas = np.empty(N)
    opt_deltas[:N-1] = best_x
    opt_deltas[N-1] = 2 * np.pi - np.sum(best_x)
    
    # Ensure all deltas are positive (safety check)
    if np.any(opt_deltas <= 0):
        # Fallback to uniform if something went wrong
        opt_deltas = np.full(N, 2 * np.pi / N)
        
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
