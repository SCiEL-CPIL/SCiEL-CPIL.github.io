import numpy as np
from scipy.optimize import minimize

N = 50

def _autoconv_peak(h):
    """
    Compute the supremum norm of the autoconvolution of a piecewise-constant
    function defined by heights h on N equal intervals over [-1/4, 1/4].
    
    Uses vectorized operations to compute the peak efficiently.
    """
    w = 1.0 / N
    # Centers of the intervals
    centers = -0.25 + (np.arange(N) + 0.5) * w
    
    # All pairwise sums of centers (candidate peak locations)
    # Shape: (N, N)
    sum_centers = centers[:, None] + centers[None, :]
    
    # Pairwise differences between all candidate locations and all pair centers
    # We need |sum_centers[p,q] - sum_centers[i,j]| for all p,q,i,j
    # This creates a 4D array of shape (N, N, N, N)
    
    # Reshape sum_centers to 4D for broadcasting
    s1 = sum_centers[:, :, None, None] # (N, N, 1, 1)
    s2 = sum_centers[None, None, :, :] # (1, 1, N, N)
    
    diff = np.abs(s1 - s2)
    
    # Triangular kernel
    tri = np.maximum(0, 1 - diff / w)
    
    # Compute values at each candidate peak location (p, q)
    H = np.outer(h, h) # (N, N)
    
    # vals[p, q] = sum_{i, j} H[i, j] * tri[p, q, i, j]
    vals = np.sum(H[None, None, :, :] * tri, axis=(2, 3))
    
    return np.max(vals)

def construct():
    # Objective function to minimize: - (||h||_2^2 / (||h||_1 * ||h*h||_inf))
    def objective(h_flat):
        h = np.maximum(0, h_flat)  # Enforce non-negativity
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 1e6
        l2_sq = np.sum(h**2)
        peak = _autoconv_peak(h)
        if peak < 1e-12:
            return 1e6
        ratio = l2_sq / (l1 * peak)
        return -ratio
    
    # Initial guess: uniform distribution
    h0 = np.ones(N) / N
    
    # Bounds: [0, inf)
    bounds = [(0, None)] * N
    
    # Use Trust Region Reflective (trf) method
    try:
        res = minimize(objective, h0, method='trf', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'xtol': 1e-10})
        h_opt = np.maximum(0, res.x)
    except Exception:
        # Fallback to L-BFGS-B if trf fails
        try:
            res = minimize(objective, h0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-8})
            h_opt = np.maximum(0, res.x)
        except Exception:
            h_opt = h0
            
    # Normalize to ensure non-negative and reasonable scale
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt / s
    else:
        h_opt = h0
        
    return h_opt
