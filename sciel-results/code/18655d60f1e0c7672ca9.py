import numpy as np
from scipy.optimize import differential_evolution

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
    # We restrict the range to ensure the sum doesn't exceed 2*pi easily
    # If sum(x) > 2*pi, the last gap is negative, and we return 0 (penalty).
    # To help DE, we set tighter bounds.
    # Average gap is 2*pi/N. Let's allow up to 2x average.
    max_gap = 4 * np.pi / N
    bounds = [(1e-5, max_gap)] * (N-1)
    
    # Use Differential Evolution for global search
    # This is slower but more robust for finding the global optimum
    # If the regular polygon is the optimum, DE will find it.
    # If there is a better non-regular config, DE will find it.
    
    result = differential_evolution(
        objective, 
        bounds,
        maxiter=200, 
        popsize=15,
        tol=1e-8,
        seed=42,
        polish=False # Polish with Nelder-Mead later if needed, or just rely on DE
    )
    
    opt_deltas = np.empty(N)
    opt_deltas[:N-1] = result.x
    opt_deltas[N-1] = 2 * np.pi - np.sum(result.x)
    
    # Fallback check: if DE failed or gave invalid result, use regular polygon
    if opt_deltas[N-1] <= 0 or np.any(opt_deltas <= 0):
        opt_deltas = np.full(N, 2 * np.pi / N)
        
    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
