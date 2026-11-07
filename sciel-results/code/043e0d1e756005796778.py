"""Symbolic regression with a mixed Chebyshev + radial-basis-let basis."""
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using 10 parameters.
    
    Basis design:
    - x is in [0.1, 6]. Map to t = (x - 3.05) / 2.95 -> [-1, 1].
    - First 6 basis functions: Chebyshev polynomials T_0 .. T_5 for smooth low-order structure.
    - Next 4 basis functions: localized radial bumps at quartile positions t = -0.5, -0.1667, 0.1667, 0.5,
      with width 0.35. These are smooth (Gaussian-like) and bounded, providing local flexibility
      for non-polynomial features while maintaining a smooth, well-conditioned landscape for BFGS.
    
    All basis functions are smooth and bounded, giving BFGS a smooth landscape.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [0.1, 6] to [-1, 1]
    t = (x - 3.05) / 2.95
    
    n = len(x)
    
    # Pre-allocate basis matrix
    B = np.empty((10, n), dtype=np.float64)
    
    # Chebyshev polynomials T_0 through T_5
    B[0, :] = 1.0
    B[1, :] = t
    T_prev = B[0, :]  # T_0
    T_curr = t        # T_1
    B[2, :] = 2.0 * t * T_curr - T_prev
    T_prev, T_curr = T_curr, B[2, :]
    B[3, :] = 2.0 * t * T_curr - T_prev
    T_prev, T_curr = T_curr, B[3, :]
    B[4, :] = 2.0 * t * T_curr - T_prev
    T_prev, T_curr = T_curr, B[4, :]
    B[5, :] = 2.0 * t * T_curr - T_prev
    
    # 4 radial basis bumps at t = -0.5, -1/6, 1/6, 0.5 with width 0.35
    centers = np.array([-0.5, -1.0/6.0, 1.0/6.0, 0.5])
    width = 0.35
    for i, c in enumerate(centers):
        B[6 + i, :] = np.exp(-((t - c) ** 2) / (2.0 * width * width))
    
    # Linear combination
    pred = B.T @ params
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
