import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Low-degree Legendre polynomials for global trend + 
    Trigonometric terms for periodic components.
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of 10 coefficients.
            params[0:4] -> Legendre P_0 to P_3 (on t = x/3)
            params[4:6] -> sin(x), cos(x)
            params[6:8] -> sin(2x), cos(2x)
            params[8:10]-> sin(3x), cos(3x)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.ravel()
        
    # Ensure finite params to avoid NaN propagation
    p = np.asarray(params, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 0.0)
    
    N = len(x)
    
    # 1. Map x to [-1, 1] for Legendre polynomials
    t = x / 3.0
    
    # 2. Compute Legendre Polynomials P_0 to P_3
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = (3t^2 - 1) / 2
    # P_3(t) = (5t^3 - 3t) / 2
    
    P0 = np.ones_like(x)
    P1 = t
    P2 = (3.0 * t * t - 1.0) * 0.5
    P3 = (5.0 * t * t * t - 3.0 * t) * 0.5
    
    # 3. Compute Trigonometric Terms
    # Using x directly (range [-3, 3])
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)
    sin_3x = np.sin(3.0 * x)
    cos_3x = np.cos(3.0 * x)
    
    # 4. Construct Basis Matrix (N, 10)
    # Order: P0, P1, P2, P3, sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x)
    V = np.column_stack([
        P0, P1, P2, P3,
        sin_x, cos_x,
        sin_2x, cos_2x,
        sin_3x, cos_3x
    ])
    
    # 5. Linear Combination
    result = V @ p
    
    # Safety check for non-finite results
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
