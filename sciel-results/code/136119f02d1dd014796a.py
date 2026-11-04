import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # We optimize the angular gaps between consecutive points on a circle.
    # There are N gaps, but they sum to 2*pi, so we have N-1 free variables.
    
    # Initial guess: uniform distribution
    # Add small random perturbation to avoid symmetry issues if any, 
    # though Nelder-Mead handles symmetry well.
    delta_init = np.full(N-1, 2 * np.pi / N)
    
    def objective(x):
        # x: array of first N-1 gaps
        # Check for invalid gaps (must be positive)
        if np.any(x <= 0):
            return 1e10 # Large penalty
            
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        
        # Check if the last gap is valid
        if deltas[N-1] <= 0:
            return 1e10
            
        # Calculate cumulative angles (starting at 0)
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Calculate the minimum triangle area numerator
        # We need min over all i < j < k of Area(i,j,k)
        # Area(i,j,k) = 0.5 * | sin(phi[j]-phi[i]) + sin(phi[k]-phi[j]) + sin(phi[i]-phi[k]+2*pi) |
        # Since phi is sorted 0 <= phi < 2*pi, and i<j<k, the angles are in order.
        # Let a = phi[j]-phi[i], b = phi[k]-phi[j], c = 2*pi - (a+b).
        # Area = 0.5 * (sin(a) + sin(b) + sin(c))
        
        # Vectorized calculation of all triple areas
        # Indices i, j, k
        i_idx, j_idx, k_idx = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
        # Filter i < j < k
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_vals = i_idx[mask]
        j_vals = j_idx[mask]
        k_vals = k_idx[mask]
        
        a = phi[j_vals] - phi[i_vals]
        b = phi[k_vals] - phi[j_vals]
        c = 2 * np.pi - a - b
        
        # sin(c) is sin(2*pi - a - b) = -sin(a+b) = -(sin(a)cos(b) + cos(a)sin(b))
        # But np.sin(c) is fine.
        areas = 0.5 * (np.sin(a) + np.sin(b) + np.sin(c))
        
        min_area = np.min(areas)
        
        # Calculate hull area denominator
        # Hull area is sum of triangles from origin to edges?
        # For points on circle, Hull Area = 0.5 * sum(sin(deltas))
        den = 0.5 * np.sum(np.sin(deltas))
        
        if den <= 0:
            return 1e10
            
        ratio = min_area / den
        return -ratio # We minimize the negative ratio to maximize the ratio

    # Bounds for each gap
    bounds = [(1e-6, 2 * np.pi - 1e-6)] * (N-1)
    
    # Use Nelder-Mead
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
    
    if result.success or (result.fun < -1e-5): # Check if we got a reasonable result
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = result.x
        opt_deltas[N-1] = 2 * np.pi - np.sum(result.x)
        
        # Ensure positivity
        if np.any(opt_deltas <= 0):
             opt_deltas = np.full(N, 2 * np.pi / N)
    else:
        # Fallback to uniform if optimization fails
        opt_deltas = np.full(N, 2 * np.pi / N)
        
    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
