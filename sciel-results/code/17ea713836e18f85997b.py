import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # Precompute indices for all triples (i, j, k) with i < j < k
    # This allows vectorized calculation of all triangle areas
    i_idx, j_idx, k_idx = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    # Initial guess: uniform distribution (regular polygon)
    delta_init = np.full(N-1, 2 * np.pi / N)
    
    def objective(x):
        # x: array of first N-1 gaps
        if np.any(x <= 0):
            return 1e6 # Large penalty
            
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        
        if deltas[N-1] <= 0:
            return 1e6
            
        # Calculate cumulative angles
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Vectorized arc calculations
        a = phi[j_idx] - phi[i_idx]
        b = phi[k_idx] - phi[j_idx]
        c = 2 * np.pi - a - b
        
        # Area numerator (2 * Area)
        nums = np.sin(a) + np.sin(b) + np.sin(c)
        min_num = np.min(nums)
        
        # Hull area denominator (2 * Hull Area)
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 1e6
            
        ratio = min_num / den
        return -ratio

    # Bounds for each gap
    bounds = [(1e-6, 2 * np.pi - 1e-6)] * (N-1)
    
    # Use Nelder-Mead with adaptive step size
    # Start from regular polygon
    result = minimize(
        objective, 
        delta_init, 
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
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = result.x
        opt_deltas[N-1] = 2 * np.pi - np.sum(result.x)
    else:
        # Fallback to uniform if optimization fails
        opt_deltas = np.full(N, 2 * np.pi / N)
        
    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
