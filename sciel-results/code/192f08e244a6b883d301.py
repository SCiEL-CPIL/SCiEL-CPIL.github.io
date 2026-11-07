import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 4 Chebyshev polynomials (global smooth approximation)
    + 6 adaptive tanh units (local nonlinearity).
    
    The Chebyshev terms (T0..T3 on t=x/3) provide a stable global polynomial
    backbone. The 6 tanh units with centers spread across [-3, 3] and varying
    scales provide localized nonlinear flexibility to capture features that
    a pure polynomial cannot.
    
    params (length 10):
      params[0..3]: Chebyshev coefficients (T0, T1, T2, T3)
      params[4..9]: tanh unit weights
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure exactly 10 params
    if params.shape[0] != 10:
        p = np.ones(10)
        n = min(10, params.shape[0])
        p[:n] = params[:n]
        params = p
    
    # Map x to [-1, 1] for Chebyshev
    t = x / 3.0
    
    # Chebyshev polynomials T0..T3 via recurrence
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    
    # 6 tanh units with fixed centers and scales
    # Centers spread across the domain, scales chosen for good coverage
    centers = np.array([-2.5, -1.25, -0.5, 0.5, 1.25, 2.5])
    scales = np.array([1.5, 1.2, 1.0, 1.0, 1.2, 1.5])
    
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3)
    
    for i in range(6):
        y += params[4 + i] * np.tanh(scales[i] * (x - centers[i]))
    
    # Ensure finite values
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
