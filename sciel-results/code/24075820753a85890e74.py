import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # We optimize the angular gaps between consecutive points on a circle.
    # There are N gaps, but they sum to 2*pi, so we have N-1 free variables.
    
    def objective(x):
        # x: array of first N-1 gaps
        # Check for invalid gaps (must be positive)
        if np.any(x <= 1e-12):
            return 1e10 # Large penalty
            
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        
        # Check if the last gap is valid
        if deltas[N-1] <= 1e-12:
            return 1e10
            
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
                    a = phi[j] - phi[i]
                    b = phi[k] - phi[j]
                    c = 2 * np.pi - a - b
                    
                    val = np.sin(a) + np.sin(b) + np.sin(c)
                    if val < min_num:
                        min_num = val
        
        # Calculate hull area denominator
        den = np.sum(np.sin(deltas))
        
        if den <= 1e-12:
            return 1e10
            
        ratio = min_num / den
        return -ratio # We minimize the negative ratio to maximize the ratio

    # Bounds for each gap
    bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
    
    best_result = None
    best_score = -np.inf
    
    # Multi-start strategy to avoid local minima
    # 1. Uniform start
    starts = [np.full(N-1, 2 * np.pi / N)]
    
    # 2. Random perturbations of uniform
    np.random.seed(42)
    for _ in range(5):
        perturb = np.random.uniform(0.9, 1.1, size=N-1)
        start = (2 * np.pi / N) * perturb
        # Normalize to sum to 2*pi - small_epsilon to keep last gap positive
        s = np.sum(start)
        start = start * (2 * np.pi * 0.99) / s
        starts.append(start)
    
    for start in starts:
        try:
            result = minimize(
                objective, 
                start, 
                method='Nelder-Mead', 
                bounds=bounds,
                options={
                    'xatol': 1e-12, 
                    'fatol': 1e-12, 
                    'maxiter': 5000, 
                    'adaptive': True
                }
            )
            if result.success:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_result = result
        except Exception:
            continue
            
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
