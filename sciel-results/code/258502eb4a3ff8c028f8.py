import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a normalized cubic spline basis.
    
    The basis consists of:
    1. Polynomial terms: 1, x, x^2, x^3
    2. Truncated power terms: (x - t_i)_+^3 for 6 internal knots t_i
    
    The basis functions are normalized by their maximum absolute value on [-2, 2]
    to improve the conditioning of the linear system for BFGS optimization.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params has length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    # Define 6 internal knots evenly spaced in [-2, 2]
    K = 6
    # Knots: -2 + (i+1)*4/7 for i=0..5
    knots = -2.0 + np.arange(1, K + 1) * (4.0 / (K + 1))
    
    # Precompute normalization constants (max abs value on [-2, 2])
    max_vals = np.zeros(10)
    max_vals[0] = 1.0   # 1
    max_vals[1] = 2.0   # x
    max_vals[2] = 4.0   # x^2
    max_vals[3] = 8.0   # x^3
    
    for i in range(K):
        t = knots[i]
        # Max of (x - t)_+^3 on [-2, 2] is at x=2
        max_vals[4 + i] = (2.0 - t) ** 3
        
    # Safety check for division
    max_vals = np.where(max_vals > 1e-12, max_vals, 1.0)
    
    # Build basis matrix
    N = len(x)
    basis = np.zeros((N, 10))
    
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x**2
    basis[:, 3] = x**3
    
    for i in range(K):
        t = knots[i]
        diff = x - t
        diff_pos = np.maximum(diff, 0.0)
        basis[:, 4 + i] = diff_pos**3
        
    # Normalize columns
    basis = basis / max_vals[np.newaxis, :]
    
    # Compute prediction
    return basis @ params
